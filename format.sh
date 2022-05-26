python -m pytest tests
black sccca --line-length 120
isort sccca --profile black
flake8 sccca --ignore E501
# poetry run bandit .
# poetry run safety check