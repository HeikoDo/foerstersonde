from streamlit import bootstrap

real_script = "foerstersonde.py"
bootstrap.run(real_script, f"run.py {real_script}", [], {})
