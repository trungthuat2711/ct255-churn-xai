"""Các mô hình đơn giản cho bài toán Churn + ghi lại loss/acc theo từng epoch (hoặc từng vòng boosting/cây)."""
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_CHOICES = ["logreg", "dtree", "rforest", "xgboost", "mlp", "svm", "knn"]
MODEL_FULLNAME = {"logreg": "Logistic Regression", "dtree": "Decision Tree", "rforest": "Random Forest",
                  "xgboost": "XGBoost", "mlp": "MLP (Neural Net)", "svm": "SVM (RBF)", "knn": "k-NN"}


def resolve_scale_pos_weight(args, y_train):
    if args.scale_pos_weight is not None:
        return float(args.scale_pos_weight)
    return float((y_train == 0).sum() / max((y_train == 1).sum(), 1))


def build_estimator(name, args, y_train):
    """Trả về estimator sklearn/xgboost chưa fit. class_weight chỉ áp dụng khi --imbalance class_weight."""
    cw = "balanced" if args.imbalance == "class_weight" else None
    seed = args.random_state
    if name == "logreg":
        return LogisticRegression(C=args.C, class_weight=cw, max_iter=args.epochs, random_state=seed,
                                  solver="lbfgs", warm_start=True)
    if name == "dtree":
        return DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                                      class_weight=cw, random_state=seed)
    if name == "rforest":
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf, class_weight=cw,
                                      random_state=seed, n_jobs=args.n_jobs, warm_start=True)
    if name == "xgboost":
        spw = resolve_scale_pos_weight(args, y_train) if args.imbalance == "class_weight" else 1.0
        return XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                             learning_rate=args.learning_rate, subsample=args.subsample,
                             colsample_bytree=args.colsample_bytree, reg_lambda=args.reg_lambda,
                             scale_pos_weight=spw, random_state=seed, n_jobs=args.n_jobs,
                             tree_method="hist", eval_metric=["logloss", "error"])
    if name == "mlp":
        hs = tuple(int(h) for h in str(args.hidden_sizes).split(",") if h)
        return MLPClassifier(hidden_layer_sizes=hs, activation="relu", solver="adam",
                             learning_rate_init=args.mlp_lr, batch_size=args.batch_size,
                             alpha=args.reg_lambda * 1e-3, random_state=seed)
    if name == "svm":
        return SVC(C=args.C, kernel="rbf", gamma="scale", probability=True, class_weight=cw, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=args.k_neighbors, weights="distance", n_jobs=args.n_jobs)
    raise ValueError(name)


def _hist():
    return {"x": [], "x_label": "epoch", "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def _record(h, x, model, Xtr, ytr, Xva, yva):
    ptr = np.clip(model.predict_proba(Xtr)[:, 1], 1e-7, 1 - 1e-7)
    h["x"].append(x)
    h["train_loss"].append(float(log_loss(ytr, ptr)))
    h["train_acc"].append(float(accuracy_score(ytr, ptr >= 0.5)))
    if len(yva):
        pva = np.clip(model.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7)
        h["val_loss"].append(float(log_loss(yva, pva)))
        h["val_acc"].append(float(accuracy_score(yva, pva >= 0.5)))


def fit_with_history(name, model, Xtr, ytr, Xva, yva, args):
    """Huấn luyện và trả (model, history, train_time_s). history = loss/acc theo epoch/iteration."""
    t0 = time.time()
    h = _hist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "mlp":
            h["x_label"] = "epoch"
            classes = np.array([0, 1])
            rng = np.random.RandomState(args.random_state)
            for ep in range(1, args.epochs + 1):
                idx = rng.permutation(len(Xtr))                       # xáo trộn mỗi epoch
                model.partial_fit(Xtr.values[idx], ytr.values[idx], classes=classes)
                _record(h, ep, model, Xtr, ytr, Xva, yva)
        elif name == "logreg":
            h["x_label"] = "L-BFGS iteration"
            for it in range(1, args.epochs + 1):
                model.set_params(max_iter=1)
                model.fit(Xtr, ytr)
                _record(h, it, model, Xtr, ytr, Xva, yva)
                if it > 5 and abs(h["train_loss"][-1] - h["train_loss"][-2]) < 1e-9:
                    break
        elif name == "xgboost":
            h["x_label"] = "boosting round"
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr)] + ([(Xva, yva)] if len(yva) else []), verbose=False)
            ev = model.evals_result()
            h["x"] = list(range(1, len(ev["validation_0"]["logloss"]) + 1))
            h["train_loss"] = ev["validation_0"]["logloss"]
            h["train_acc"] = [1 - e for e in ev["validation_0"]["error"]]
            if len(yva):
                h["val_loss"] = ev["validation_1"]["logloss"]
                h["val_acc"] = [1 - e for e in ev["validation_1"]["error"]]
        elif name == "rforest":
            h["x_label"] = "number of trees"
            final_n = args.n_estimators
            step = max(1, final_n // 20)
            for n in list(range(step, final_n + 1, step)) + ([final_n] if final_n % step else []):
                model.set_params(n_estimators=n)
                model.fit(Xtr, ytr)
                _record(h, n, model, Xtr, ytr, Xva, yva)
        elif name == "dtree":
            h["x_label"] = "max_depth (sweep)"
            for d in range(1, max(args.max_depth + 6, 12) + 1):
                tmp = DecisionTreeClassifier(max_depth=d, min_samples_leaf=args.min_samples_leaf,
                                             class_weight=model.class_weight, random_state=args.random_state)
                tmp.fit(Xtr, ytr)
                _record(h, d, tmp, Xtr, ytr, Xva, yva)
            model.fit(Xtr, ytr)
        else:                                                           # svm, knn: không có quá trình lặp
            model.fit(Xtr, ytr)
            h = None
    return model, h, time.time() - t0


def describe_architecture(name, model, n_features):
    arch = {"model_type": MODEL_FULLNAME[name], "sklearn_class": type(model).__name__, "n_input_features": n_features}
    if name == "logreg":
        arch.update({"n_parameters": int(model.coef_.size + model.intercept_.size), "n_iter_run": int(np.max(model.n_iter_))})
    elif name == "dtree":
        arch.update({"depth": int(model.get_depth()), "n_leaves": int(model.get_n_leaves()), "node_count": int(model.tree_.node_count)})
    elif name == "rforest":
        arch.update({"n_trees": len(model.estimators_),
                     "mean_tree_depth": float(np.mean([e.get_depth() for e in model.estimators_])),
                     "total_nodes": int(sum(e.tree_.node_count for e in model.estimators_))})
    elif name == "xgboost":
        b = model.get_booster()
        arch.update({"n_boosted_rounds": int(b.num_boosted_rounds()), "booster": "gbtree",
                     "objective": "binary:logistic", "tree_method": "hist"})
    elif name == "mlp":
        sizes = [n_features] + list(model.hidden_layer_sizes) + [1]
        arch.update({"layer_sizes": sizes, "hidden_activation": model.activation, "output_activation": "logistic",
                     "optimizer": model.solver, "n_parameters": int(sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)),
                     "epochs_run": int(model.n_iter_)})
    elif name == "svm":
        arch.update({"kernel": "rbf", "n_support_vectors": [int(v) for v in model.n_support_]})
    elif name == "knn":
        arch.update({"n_neighbors": model.n_neighbors, "weights": model.weights})
    return arch
