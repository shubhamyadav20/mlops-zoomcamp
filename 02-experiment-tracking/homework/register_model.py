import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Experiment names
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"

# Parameters to log
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    """Load a pickle file"""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    """Train and evaluate a model with given params, log metrics and model to MLflow"""
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {param: int(params[param]) for param in RF_PARAMS}
        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))

        # Log metrics
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the trained model
        mlflow.sklearn.log_model(rf, artifact_path="model")

        print(f"Run {mlflow.active_run().info.run_id} - val_rmse: {val_rmse:.3f}, test_rmse: {test_rmse:.3f}")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    """Evaluate top N hyperparameter runs, retrain, and register the best model."""
    client = MlflowClient()

    # Get the best N runs from the hyperparameter optimization experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    print(f"Evaluating top {len(runs)} models from experiment '{HPO_EXPERIMENT_NAME}'...")
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Get the new experiment where retrained models were logged
    best_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=best_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name="RandomForestRegressorModel")

    print("\n=== Best Model Summary ===")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Test RMSE: {best_run.data.metrics['test_rmse']:.3f}")
    print(f"Model registered as: {result.name}")
    print(f"Version: {result.version}")


if __name__ == '__main__':
    run_register_model()
