import subprocess

# Define your models and whether they are masked or autoregressive
models_info = [
    {"name": "german_bert-finetuned", "masked": True},
    {"name": "multi_bert-finetuned", "masked": True},
    {"name": "xlm-roberta-base-finetuned", "masked": True},
    {"name": "gpt2-finetuned", "masked": False},
    {"name": "bloom-560m-finetuned", "masked": False},
    {"name": "opt-350m-finetuned", "masked": False},
    {"name": "xlm-clm-ende-1024-finetuned", "masked": False},
    {"name": "xlm-mlm-ende-1024-finetuned", "masked": True},
    {"name": "german_bert", "masked": True},
    {"name": "multi_bert", "masked": True},
    {"name": "xlm-roberta-base", "masked": True},
    {"name": "gpt2", "masked": False},
    {"name": "bloom-560m", "masked": False},
    {"name": "opt-350m", "masked": False},
    {"name": "xlm-clm-ende-1024", "masked": False},
    {"name": "xlm-mlm-ende-1024", "masked": True},
]

input_file = "winoqueerDE.csv"
output_template = "results_{model_name}_{script}.csv"

for model_info in models_info:
    model_name = model_info["name"]
    masked = model_info["masked"]
    model_safe = model_name.replace("/", "_")

    # Run metric.py only for masked models
    if masked:
        output_file = output_template.format(model_name=model_safe, script="metric")
        cmd_metric = [
            "python", "metric.py",
            "--input_file", input_file,
            "--lm_model_path", model_name,
            "--output_file", output_file,
        ]
        print("Running metric.py:", " ".join(cmd_metric))
        subprocess.run(cmd_metric)
        
    else:
        output_file = output_template.format(model_name=model_safe, script="autoregressive")
        cmd_autoregressive = [
            "python", "metric_autoregressive.py",
            "--input_file", input_file,
            "--lm_model_path", model_name,
            "--output_file", output_file,
        ]
        print("Running metric_autoregressive.py:", " ".join(cmd_autoregressive))
        subprocess.run(cmd_autoregressive)

    # Run metricSurvey.py for both masked and autoregressive
    """input_file_survey = "UmfrageWinoQueer.csv"
    output_file = output_template.format(model_name=model_safe, script="survey")
    cmd_survey = [
        "python", "metricSurvey.py",
        "--input_file", input_file_survey,
        "--lm_model_path", model_name,
        "--output_file", output_file,
    ]
    if masked:
        cmd_survey.append("--masked")"""
    
    print("Running metricSurvey.py:", " ".join(cmd_survey))
    subprocess.run(cmd_survey)
