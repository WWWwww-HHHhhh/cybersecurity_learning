"""固定条件计时及输入组合隔离划分的重复验证。"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import gc
import os
import sys
import shutil
import platform
import subprocess
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, matthews_corrcoef, confusion_matrix
from threadpoolctl import threadpool_limits, threadpool_info

BINARY = ['dns_query','ssl_version','ssl_cipher','ssl_subject','ssl_issuer','http_uri','http_user_agent','http_orig_mime_types','http_resp_mime_types','weird_name']
CATEGORY = ['proto','service','conn_state','dns_AA','dns_RD','dns_RA','dns_rejected','ssl_resumed','ssl_established','http_trans_depth','http_method','http_version','weird_addl','weird_notice']
METRICS = ['Macro-F1','攻击召回率','攻击精确率','Accuracy','Weighted-F1','MCC']
TIMES = ['训练准备秒','测试转换秒','模型训练秒','模型预测秒','训练总秒','测试总秒']

def save_table(table, path):
    table.to_csv(path, index=False, encoding='utf-8-sig')
    converters = {c: str for c in table if pd.api.types.is_string_dtype(table[c].dtype) or table[c].dtype == object}
    restored = pd.read_csv(path, converters=converters)
    pd.testing.assert_frame_equal(restored, table.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-10, atol=1e-12)

def save_arrays(path, payload):
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(payload)
        for key, values in payload.items(): np.testing.assert_array_equal(data[key], values)

def summary(table, groups, columns):
    rows=[]
    for key, block in table.groupby(groups, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in columns:
            v=block[metric]
            rows.append({**dict(zip(groups,key)), '指标':metric, '次数':len(v), '均值':v.mean(),
                         '样本标准差':v.std(ddof=1), '最小值':v.min(), '最大值':v.max()})
    return pd.DataFrame(rows)

def measure(y, pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(zip(METRICS,[f1_score(y,pred,average='macro',zero_division=0),recall_score(y,pred,zero_division=0),
        precision_score(y,pred,zero_division=0),accuracy_score(y,pred),f1_score(y,pred,average='weighted',zero_division=0),
        matthews_corrcoef(y,pred)]), TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))

def estimator(name, settings, seed=42):
    p=settings['model_parameters'][name].copy();p['random_state']=seed
    return {'DT':DecisionTreeClassifier,'RF':RandomForestClassifier}[name](**p)

def pearson(train):
    active=np.ptp(train,axis=0)>0
    scores=np.full(train.shape[1],np.nan)
    if active.sum()==1: scores[active]=1.
    elif active.any(): scores[active]=np.corrcoef(train[:,active],rowvar=False).mean(axis=0)
    return scores, np.flatnonzero(~active)

def specs_for(train, names, thresholds):
    scores,constant=pearson(train)
    specs=[{'method':'Original','size':train.shape[1],'threshold':None}]
    seen=set();audit=[]
    for t in thresholds:
        pos=np.flatnonzero(np.abs(scores)<=t);sig=tuple(pos)
        status='用于配对'
        if not len(pos): status='没有入选列'
        elif len(pos)==train.shape[1]: status='全部输入，与 Original 相同'
        elif sig in seen: status='与已有集合重复'
        else:
            specs.extend([{'method':'FS','size':len(pos),'threshold':t,'positions':pos},
                          {'method':'PCA','size':len(pos),'threshold':t}]);seen.add(sig)
        audit.append({'阈值':t,'维数':len(pos),'处理':status,'入选列':json.dumps(names[pos].tolist(),ensure_ascii=False)})
    specs.append({'method':'PCA','size':train.shape[1],'threshold':None})
    return specs,audit,constant,scores

def prepare(train,test,spec):
    start=perf_counter();method=spec['method'];extra={}
    if method=='FS':
        scores,_=pearson(train)
        pos=np.flatnonzero(np.abs(scores)<=spec['threshold'])
        np.testing.assert_array_equal(pos,spec['positions'])
        tr=np.ascontiguousarray(train[:,pos]);extra={'positions':pos}
    elif method=='PCA':
        pca=PCA(n_components=spec['size'],svd_solver='full',whiten=False,copy=True)
        tr=np.ascontiguousarray(pca.fit_transform(train))
        extra={'components':pca.components_,'mean':pca.mean_,'variance_ratio':pca.explained_variance_ratio_}
    else: tr=np.ascontiguousarray(train)
    train_seconds=perf_counter()-start
    start=perf_counter()
    if method=='FS': te=np.ascontiguousarray(test[:,pos])
    elif method=='PCA': te=np.ascontiguousarray(pca.transform(test))
    else: te=np.ascontiguousarray(test)
    test_seconds=perf_counter()-start
    assert tr.shape[1]==te.shape[1]==spec['size']
    assert np.isfinite(tr).all() and np.isfinite(te).all()
    return tr,te,train_seconds,test_seconds,extra

def fit_record(tr,te,ytr,yte,model,spec,settings,a,b):
    est=estimator(model,settings)
    start=perf_counter();est.fit(tr,ytr);fit=perf_counter()-start
    start=perf_counter();pred=est.predict(te).astype(np.int8);predict=perf_counter()-start
    row={'模型':model,'输入方案':spec['method'],'维数':spec['size'],'Pearson阈值':spec['threshold'],
         **measure(yte,pred),'训练准备秒':a,'测试转换秒':b,'模型训练秒':fit,'模型预测秒':predict,
         '训练总秒':a+fit,'测试总秒':b+predict}
    return row,pred

def run_timing(root,settings,repeats=5):
    out=root/'results/04_validation'
    for folder in ['tables','arrays','metadata']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    with np.load(root/settings['input_file'],allow_pickle=False) as z:
        train=z['group_X_train'];test=z['group_X_test'];ytr=z['group_y_train'];yte=z['group_y_test']
        names=z['group_feature_names'];indices=z['group_test_index'];types=z['group_test_type']
    with np.load(root/'results/03_features_and_models/arrays/test_predictions.npz',allow_pickle=False) as z:
        original={k:z[k] for k in z.files if k not in ['y_test','test_index','test_type']}
    with threadpool_limits(limits=4): specs,_,_,_=specs_for(train,names,settings['thresholds'])
    rows=[];payload={'y_test':yte,'test_index':indices,'test_type':types};ref={}
    for repeat in range(1,repeats+1):
        order=np.random.default_rng(1000+repeat).permutation(len(specs))
        for position in order:
            spec=specs[position]
            with threadpool_limits(limits=4):
                tr,te,a,b,_=prepare(train,test,spec)
                model_order=['DT','RF'] if repeat%2 else ['RF','DT']
                for model in model_order:
                    row,pred=fit_record(tr,te,ytr,yte,model,spec,settings,a,b)
                    key=f"{model}_{spec['method']}_{spec['size']}"
                    if key in ref: np.testing.assert_array_equal(pred,ref[key])
                    else: ref[key]=pred.copy()
                    np.testing.assert_array_equal(pred,original[key])
                    rows.append({'重复轮次':repeat,'执行顺序':int(np.flatnonzero(order==position)[0])+1,**row})
                    payload[f'{key}_repeat{repeat}']=pred
            del tr,te
        print(f'完成固定条件重复：{repeat}/{repeats}',flush=True)
    result=pd.DataFrame(rows)
    aggregate=summary(result,['模型','输入方案','维数'],METRICS+TIMES)
    pairs=[]
    for (repeat,model),block in result.groupby(['重复轮次','模型']):
        base=block[block['输入方案']=='Original'].iloc[0]
        for _,r in block[block['输入方案']!='Original'].iterrows():
            pairs.append({'重复轮次':repeat,'模型':model,'输入方案':r['输入方案'],'维数':r['维数'],
                          **{f'相对Original的{m}变化':r[m]-base[m] for m in ['Macro-F1','攻击召回率','训练总秒','测试总秒']}})
    pairs=pd.DataFrame(pairs)
    for name,t in [('timing_results',result),('timing_summary',aggregate),('timing_pairs',pairs)]:save_table(t,out/f'tables/{name}.csv')
    save_arrays(out/'arrays/timing_predictions.npz',payload)
    return result,aggregate,pairs

def run_resplits(root,settings,seeds):
    out=root/'results/04_validation'
    for folder in ['tables','arrays','metadata']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    path=root/'data/raw/Train_Test_Network.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['raw_sha256']
    df=pd.read_csv(path,low_memory=False).drop_duplicates().copy()
    X=df.drop(columns=['ts','src_ip','dst_ip','src_port','dst_port','label','type']).copy()
    number=[c for c in X if c not in BINARY+CATEGORY]
    X[BINARY+CATEGORY]=X[BINARY+CATEGORY].replace('-','n/a')
    for c in BINARY:X[c]=X[c].ne('n/a').astype('int8')
    assert X.shape==(449972,38) and not X.isna().any().any()
    y=df['label'];groups=X.groupby(list(X.columns),sort=False,dropna=False).ngroup()
    codes,type_names=pd.factorize(df['type']);share=np.bincount(codes)/len(df)
    rows=[];audits=[];candidates=[];types_rows=[];threshold_rows=[];unseen_rows=[];constant_rows=[];pairs=[];overlap_rows=[]
    test_sets={}
    for seed in seeds:
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        best=None;score_best=float('inf')
        for fold,(trpos,tepos) in enumerate(cv.split(X,df['type'],groups)):
            dist=.5*np.abs(np.bincount(codes[tepos],minlength=len(type_names))/len(tepos)-share).sum()
            score=4*abs(len(tepos)/len(X)-.2)+4*abs(y.iloc[tepos].mean()-y.mean())+4*dist
            candidates.append({'划分种子':seed,'候选折':fold,'测试比例':len(tepos)/len(X),'攻击比例':y.iloc[tepos].mean(),'类型总变差':dist,'得分':score})
            if score<score_best: score_best=score;best=(fold,trpos,tepos,dist)
        fold,trpos,tepos,dist=best
        assert not set(groups.iloc[trpos])&set(groups.iloc[tepos])
        train_raw,test_raw=X.iloc[trpos],X.iloc[tepos]
        encoder=ColumnTransformer([('number','passthrough',number),('binary','passthrough',BINARY),
            ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64),CATEGORY)],
            remainder='drop',sparse_threshold=0,verbose_feature_names_out=False)
        train_encoded=encoder.fit_transform(train_raw).astype(np.float64,copy=False)
        test_encoded=encoder.transform(test_raw).astype(np.float64,copy=False)
        names=encoder.get_feature_names_out();scaler=MinMaxScaler(copy=True)
        train=scaler.fit_transform(train_encoded);test=scaler.transform(test_encoded)
        del train_encoded,test_encoded
        trhash=pd.util.hash_pandas_object(pd.DataFrame(train),index=False)
        tehash=pd.util.hash_pandas_object(pd.DataFrame(test),index=False)
        shared=int(tehash.isin(set(trhash)).sum());assert shared==0
        del trhash,tehash
        if seed==42:
            with np.load(root/settings['input_file'],allow_pickle=False) as z:
                for a,b in [(train,z['group_X_train']),(test,z['group_X_test']),(df.index[trpos],z['group_train_index']),
                            (df.index[tepos],z['group_test_index']),(names,z['group_feature_names'])]:np.testing.assert_array_equal(a,b)
        ytr=y.iloc[trpos].to_numpy();yte=y.iloc[tepos].to_numpy();test_sets[seed]=set(df.index[tepos])
        for typ in sorted(set(df['type'])):
            types_rows.append({'划分种子':seed,'流量类型':typ,'训练行数':int(df.iloc[trpos]['type'].eq(typ).sum()),'测试行数':int(df.iloc[tepos]['type'].eq(typ).sum())})
        for col in CATEGORY:
            unseen=set(test_raw[col])-set(train_raw[col])
            for value in sorted(unseen):unseen_rows.append({'划分种子':seed,'原始列名':col,'仅测试侧取值':value,'测试行数':int(test_raw[col].eq(value).sum())})
        with threadpool_limits(limits=4):specs,ta,constant,scores=specs_for(train,names,settings['thresholds'])
        threshold_rows.extend([{'划分种子':seed,**r} for r in ta])
        constant_rows.append({'划分种子':seed,'恒定列数':len(constant),'恒定列名':json.dumps(names[constant].tolist(),ensure_ascii=False)})
        audits.append({'划分种子':seed,'选中折':fold,'训练行数':len(trpos),'测试行数':len(tepos),'测试比例':len(tepos)/len(X),
                       '训练攻击比例':ytr.mean(),'测试攻击比例':yte.mean(),'类型总变差':dist,'输入维数':len(names),
                       '共享精确输入组合数':0,'编码缩放后共享指纹测试行数':shared,'测试越界值数':int(((test<0)|(test>1)).sum())})
        payload={'train_index':df.index[trpos].to_numpy(),'test_index':df.index[tepos].to_numpy(),
                 'y_train':ytr,'y_test':yte,'test_type':df.iloc[tepos]['type'].to_numpy(dtype=str),'feature_names':names.astype(str),
                 'pearson_scores':scores,'scaler_min':scaler.min_,'scaler_scale':scaler.scale_,
                 'scaler_data_min':scaler.data_min_,'scaler_data_max':scaler.data_max_}
        category_map={col:list(map(str,values)) for col,values in zip(CATEGORY,encoder.named_transformers_['category'].categories_)}
        (out/f'metadata/split_{seed}_categories.json').write_text(json.dumps(category_map,ensure_ascii=False,indent=2)+'\n')
        local=[]
        for spec in specs:
            suffix=('full' if spec['threshold'] is None else str(spec['threshold']).replace('.','p'))
            tag=f"{spec['method']}_{suffix}"
            with threadpool_limits(limits=4):
                tr,te,a,b,extra=prepare(train,test,spec)
                for k,v in extra.items():payload[f'{tag}_{k}']=v
                for model in ['DT','RF']:
                    row,pred=fit_record(tr,te,ytr,yte,model,spec,settings,a,b)
                    row={'划分种子':seed,'模型随机种子':42,'候选':('完整输入' if spec['threshold'] is None else f"阈值 {spec['threshold']}"),**row};rows.append(row);local.append(row)
                    payload[f'{model}_{tag}_pred']=pred
                    if seed==42:
                        with np.load(root/'results/03_features_and_models/arrays/test_predictions.npz',allow_pickle=False) as z:
                            np.testing.assert_array_equal(pred,z[f"{model}_{spec['method']}_{spec['size']}"])
            del tr,te
        save_arrays(out/f'arrays/split_{seed}_artifacts.npz',payload)
        local=pd.DataFrame(local)
        for model in ['DT','RF']:
            block=local[local['模型']==model]
            for _,r in block[block['输入方案']=='PCA'].iterrows():
                comparison='Original' if r['候选']=='完整输入' else 'FS'
                base=block[(block['输入方案']==comparison)&(block['候选']==r['候选'])].iloc[0]
                pairs.append({'划分种子':seed,'模型':model,'候选':r['候选'],'维数':r['维数'],'比较':f'PCA减{comparison}',
                              **{m+'差值':r[m]-base[m] for m in METRICS}})
        print(f'完成隔离划分种子 {seed}：{len(names)} 维，{len(local)} 个实验',flush=True)
        del train,test,payload,encoder,scaler;gc.collect()
    for i,a in enumerate(seeds):
        for b in seeds[i+1:]:
            overlap_rows.append({'划分种子A':a,'划分种子B':b,'共同测试行数':len(test_sets[a]&test_sets[b]),
                                  '测试集合完全相同':test_sets[a]==test_sets[b]})
    result=pd.DataFrame(rows);pair=pd.DataFrame(pairs)
    tables={'划分检查':pd.DataFrame(audits),'全部候选折':pd.DataFrame(candidates),'全部类型数量':pd.DataFrame(types_rows),
            '阈值与入选列':pd.DataFrame(threshold_rows),'仅测试侧类别':pd.DataFrame(unseen_rows,columns=['划分种子','原始列名','仅测试侧取值','测试行数']),
            '恒定列检查':pd.DataFrame(constant_rows),'跨轮测试重叠':pd.DataFrame(overlap_rows),'逐次模型结果':result,
            '按阈值汇总':summary(result,['模型','输入方案','候选'],METRICS), '逐次配对差值':pair,
            '配对差值汇总':summary(pair,['模型','候选','比较'],[m+'差值' for m in METRICS])}
    file_names=['split_audit','split_candidates','split_types','split_features','split_unseen','split_constants',
                'split_test_overlap','split_results','split_summary','split_pairs','split_pair_summary']
    for name,table in zip(file_names,tables.values()):save_table(table,out/f'tables/{name}.csv')
    manifest={'split_seeds':seeds,'model_seed':42,'thresholds':settings['thresholds'],
              'source_input_sha256':settings['input_sha256'],'raw_sha256':settings['raw_sha256'],
              'selection':'minimum distribution deviation, never model scores','threads':4,
              'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='validation_manifest.json'}}
    (out/'metadata/validation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return tables

def capture_environment(root):
    out=root/'results/environment';out.mkdir(exist_ok=True,parents=True)
    conda_exe=os.environ.get('CONDA_EXE') or shutil.which('conda') or str(Path(sys.prefix).parents[1]/'bin/conda')
    environment={'python':platform.python_version(),'python_build':platform.python_build(),
        'platform':platform.platform(),'kernel':platform.release(),
        'os_release':Path('/etc/os-release').read_text(),'cpu':subprocess.check_output(['lscpu'],text=True),
        'memory':Path('/proc/meminfo').read_text(),'threads':{'RF_n_jobs':4,'BLAS_OpenMP_limit':4},
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'git':subprocess.check_output(['git','--version'],text=True).strip(),
        'conda':subprocess.check_output([conda_exe,'--version'],text=True).strip()}
    with threadpool_limits(limits=4):environment['active_threadpools']=threadpool_info()
    powershell=Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
    if powershell.exists():
        command = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);system=Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,HypervisorPresent;os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber} | ConvertTo-Json -Depth 5"
        environment['windows_host']=json.loads(subprocess.check_output([str(powershell),'-NoProfile','-Command',command]).decode('utf-8-sig'))
    environment['wsl_version']=subprocess.check_output(['/mnt/c/Windows/System32/wsl.exe','--version']).decode('utf-16-le',errors='replace').strip()
    (out/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
    conda_packages=json.loads(subprocess.check_output([conda_exe,'list','-p',__import__('sys').prefix,'--json'],text=True))
    (out/'conda-packages.json').write_text(json.dumps(conda_packages,ensure_ascii=False,indent=2)+'\n')
    exported=subprocess.check_output([conda_exe,'env','export','-p',__import__('sys').prefix],text=True)
    exported='\n'.join(line for line in exported.splitlines() if not line.startswith('prefix:'))+'\n'
    (out/'conda-environment.yml').write_text(exported)
    (out/'package-versions.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(environment['packages'].items()))+'\n')
    freeze=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)
    (out/'pip-freeze.txt').write_text(freeze)
    return environment
