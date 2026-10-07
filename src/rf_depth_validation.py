"""固定隔离划分及模型种子的 RF 深度敏感性检查。"""
from pathlib import Path
import hashlib
import json
import platform
from time import perf_counter
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from threadpoolctl import threadpool_limits
from experiment_validation import specs_for, prepare, measure, save_table, save_arrays, METRICS


def run_rf_depth(root):
    root = Path(root)
    source = root / 'results/03_features_and_models'
    settings_path = source / 'metadata/experiment_settings.json'
    settings = json.loads(settings_path.read_text())
    input_path = root / settings['input_file']
    source_paths = [input_path, settings_path, source/'arrays/test_predictions.npz',
                    source/'tables/fs_selected_features.csv', source/'tables/pca_loadings.csv', source/'tables/pca_variance.csv']
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert source_hashes[str(input_path.relative_to(root))] == settings['input_sha256']
    assert settings['model_parameters']['RF']['max_depth'] == 5
    assert settings['random_seed'] == 42
    with np.load(input_path, allow_pickle=False) as data:
        train, test = data['group_X_train'], data['group_X_test']
        y_train, y_test = data['group_y_train'], data['group_y_test']
        names = data['group_feature_names']
        test_index, test_types = data['group_test_index'], data['group_test_type']
        train_index = data['group_train_index']
    train_hash = hashlib.sha256(train.tobytes()).hexdigest()
    test_hash = hashlib.sha256(test.tobytes()).hexdigest()
    with np.load(source/'arrays/test_predictions.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['test_index'], test_index)
        np.testing.assert_array_equal(data['y_test'], y_test)
        old_predictions = {key: data[key] for key in data.files if key.startswith('RF_')}
    selected = pd.read_csv(source/'tables/fs_selected_features.csv')
    loadings = pd.read_csv(source/'tables/pca_loadings.csv')
    variance = pd.read_csv(source/'tables/pca_variance.csv')
    with threadpool_limits(limits=4):
        specs, _, constants, _ = specs_for(train, names, settings['thresholds'])
    assert len(constants) == 0
    assert len(specs) == 10
    assert {s['size'] for s in specs if s['method']=='FS'} == set(settings['fs_dimensions'])
    assert {s['size'] for s in specs if s['method']=='PCA'} == set(settings['pca_dimensions'])
    depths = [5, 10, None]
    parameters = {}
    parameter_rows = []
    for depth in depths:
        label = str(depth) if depth is not None else '不设上限'
        params = settings['model_parameters']['RF'].copy()
        params['max_depth'] = depth
        params['random_state'] = 42
        assert {k for k in params if params[k] != settings['model_parameters']['RF'][k]} <= {'max_depth'}
        parameters[label] = params
        parameter_rows.extend({'最大深度设置':label, '参数':key, '取值':str(value)} for key,value in params.items())
    result_rows, diagnostic_rows, attack_rows, reproduction_rows = [], [], [], []
    payload = {'y_test':y_test, 'test_index':test_index, 'test_type':test_types.astype(str), 'train_index':train_index}
    for spec in specs:
        method, size = spec['method'], spec['size']
        with threadpool_limits(limits=4):
            tr, te, prep_seconds, convert_seconds, extra = prepare(train, test, spec)
        if method == 'FS':
            saved_names = selected.loc[np.isclose(selected['阈值'], spec['threshold']), '输入列名'].to_numpy()
            np.testing.assert_array_equal(names[spec['positions']], saved_names)
        if method == 'PCA':
            components = [f'PC{i}' for i in range(1, size+1)]
            saved = loadings[loadings['方案维数']==size].pivot(index='主成分', columns='输入列名', values='组合系数').loc[components,names]
            np.testing.assert_allclose(extra['components'], saved.to_numpy(), rtol=1e-9, atol=1e-10)
            ratios = variance[variance['方案维数']==size].sort_values('主成分序号')['解释方差比例']
            np.testing.assert_allclose(extra['variance_ratio'], ratios, rtol=1e-9, atol=1e-12)
        for depth in depths:
            label = str(depth) if depth is not None else '不设上限'
            model = RandomForestClassifier(**parameters[label])
            with threadpool_limits(limits=4):
                start = perf_counter()
                model.fit(tr, y_train)
                fit_seconds = perf_counter()-start
                start = perf_counter()
                predicted = model.predict(te).astype(np.int8)
                predict_seconds = perf_counter()-start
                # 训练预测只用于诊断，不计入训练与测试耗时。
                train_predicted = model.predict(tr).astype(np.int8)
            key = {'最大深度设置':label, '输入方案':method, '维数':size}
            metrics = measure(y_test, predicted)
            result_rows.append({**key, '模型随机种子':42, 'Pearson阈值':spec['threshold'], **metrics,
                               '训练准备秒':prep_seconds, '测试转换秒':convert_seconds,
                               '模型训练秒':fit_seconds, '模型预测秒':predict_seconds,
                               '训练总秒':prep_seconds+fit_seconds, '测试总秒':convert_seconds+predict_seconds})
            train_f1 = f1_score(y_train, train_predicted, average='macro', zero_division=0)
            tree_depths = [tree.get_depth() for tree in model.estimators_]
            leaves = [tree.get_n_leaves() for tree in model.estimators_]
            assert len(tree_depths) == 100
            if depth is not None: assert max(tree_depths) <= depth
            diagnostic_rows.append({**key, '训练Macro-F1':train_f1, '测试Macro-F1':metrics['Macro-F1'],
                                    '训练减测试Macro-F1':train_f1-metrics['Macro-F1'],
                                    '训练Accuracy':accuracy_score(y_train,train_predicted),
                                    '测试Accuracy':metrics['Accuracy'], '树数量':len(tree_depths),
                                    '实际最小树深':min(tree_depths), '实际最大树深':max(tree_depths),
                                    '平均叶子数':float(np.mean(leaves))})
            old_key = f'RF_{method}_{size}'
            if depth == 5:
                differences = int(np.count_nonzero(predicted != old_predictions[old_key]))
                assert differences == 0, old_key
                reproduction_rows.append({**key, '与03预测不同的行数':differences})
            for traffic_type in sorted(set(test_types)-{'normal'}):
                mask = test_types == traffic_type
                total = int(mask.sum()); detected = int(predicted[mask].sum())
                attack_rows.append({**key, '攻击类型':traffic_type, '测试记录数':total, '检出数':detected,
                                    '漏报数':total-detected, '攻击类型召回率':detected/total})
            payload[f"RF_depth{depth}_{method}_{size}"] = predicted
            print(f'完成 RF：{method} {size} 维，最大深度 {label}', flush=True)
            del model, train_predicted
        del tr,te
    results = pd.DataFrame(result_rows)
    keys = ['最大深度设置','输入方案','维数']
    assert len(results)==30 and not results.duplicated(keys).any()
    pairs=[]
    for label in parameters:
        block = results[results['最大深度设置']==label].set_index(['输入方案','维数'])
        for size in settings['pca_dimensions']:
            comparison = 'Original' if size==len(names) else 'FS'
            a,b = block.loc[('PCA',size)], block.loc[(comparison,size)]
            pairs.append({'最大深度设置':label, '维数':size, '比较':f'PCA减{comparison}',
                          **{m+'差值':a[m]-b[m] for m in METRICS}})
    pairs = pd.DataFrame(pairs)
    changes=[]
    base = results[results['最大深度设置']=='5'].set_index(['输入方案','维数'])
    for _,row in results.iterrows():
        original = base.loc[(row['输入方案'],row['维数'])]
        changes.append({k:row[k] for k in keys} | {f'相对深度5的{m}变化':row[m]-original[m] for m in METRICS})
    tables = {'完整参数':pd.DataFrame(parameter_rows), '深度5预测核对':pd.DataFrame(reproduction_rows),
              '全部检测指标与单次耗时':results, '训练测试与树规模':pd.DataFrame(diagnostic_rows),
              '同深度配对差值':pairs, '相对深度5的分数变化':pd.DataFrame(changes),
              '全部攻击类型检出情况':pd.DataFrame(attack_rows)}
    filenames = ['parameters','reproduction','results','diagnostics','pairs','depth_changes','attack_types']
    out = root/'results/04_rf_depth'
    for folder in ['tables','arrays','metadata']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    for name,table in zip(filenames,tables.values()): save_table(table,out/f'tables/{name}.csv')
    save_arrays(out/'arrays/test_predictions.npz',payload)
    assert hashlib.sha256(train.tobytes()).hexdigest()==train_hash
    assert hashlib.sha256(test.tobytes()).hexdigest()==test_hash
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    manifest = {'kind':'rf_depth_sensitivity_fixed_split_and_seed', 'depths':depths,
                'split_seed':42, 'model_seed':42, 'run_count':len(results), 'model_parameters':parameters,
                'source_sha256':source_hashes, 'test_rows':len(y_test), 'train_rows':len(y_train),
                'threads':{'RF_n_jobs':4,'BLAS_OpenMP_limit':4}, 'timing_repeats':1,
                'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'src/rf_depth_validation.py',root/'src/experiment_validation.py']},
                'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}}
    (out/'metadata/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    assert json.loads((out/'metadata/manifest.json').read_text())==manifest
    return tables
