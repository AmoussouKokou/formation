import mlflow
from mlflow.models import infer_signature

def log_mlflow(
    model, 
    params=None, 
    metrics=None, 
    artifacts=None, 
    experiment_name="Default_Experiment", 
    run_name=None, 
    tags=None, 
    tracking_uri="http://localhost:5000",
    X=None
):
    """
    Enregistre les paramètres, métriques, modèle et artefacts dans MLflow.
    
    :param model: Modèle ML à logger
    :param params: Dictionnaire des paramètres du modèle
    :param metrics: Dictionnaire des métriques à logger
    :param artifacts: Liste des chemins des artefacts à enregistrer
    :param experiment_name: Nom de l'expérience MLflow
    :param run_name: Nom du run (optionnel)
    :param tags: Dictionnaire de tags à associer au run (optionnel)
    :param tracking_uri: URI du serveur MLflow (par défaut "http://localhost:5000") 127.0.0.1
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        
        if tags:
            mlflow.set_tags(tags)
        
        if params:
            mlflow.log_params(params)
        
        if metrics:
            mlflow.log_metrics(metrics)
        
        if model:
            mlflow.sklearn.log_model(model, "model")
        
        if artifacts:
            for artifact in artifacts:
                mlflow.log_artifact(artifact)
                
        if X is not None:
        
            # Infer the model signature
            signature = infer_signature(X, model.predict(X))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="cse_model",
                signature=signature,
                input_example=X,
                registered_model_name="tracking-quickstart",
            )
    
    print(f"Run enregistré sous l'expérience '{experiment_name}' sur {tracking_uri}")
