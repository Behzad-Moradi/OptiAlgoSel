### OptiAlgoSel
OptiAlgoSel is an AI-powered algorithm selection platform for black-box optimisation that helps practitioners identify the most suitable optimisation algorithm for a given problem using landscape analysis, machine learning, and intelligent AI assistants.

There are three main directories in this repository:

1. DataBase - This directory contains the database schema.
2. Utils - This directory contains utility functions for data manipulation and analysis.
3. Reports - This directory contains the resuls and visulisation.



### How It Works
1. **Design of Experiments (DOE) & Feature Extraction**: A user provides an initial set of sample points and their evaluated function values. OptiAlgoSel validates the bounds and sample distribution, then performs **Exploratory Landscape Analysis (ELA)** using pflacco to compute mathematical properties (dispersion, level sets, information content, curvature, and meta-model fits).
2. **Multi-Label ML Classification**: The extracted landscape features are passed to trained multi-label classifiers (such as XGBoost, Random Forest, SVM, and MLP, trained in train_test_models.py. The models are trained on extensive benchmark suites (including BBOB for continuous optimization and SMD for bi-level optimization) to identify which solvers are statistically likely to reach optimal solutions.
3. **Delivery & Service**: The core workflow is exposed through a FastAPI backend mainapi.py. When a prediction is requested via the prediction router, the platform logs the request to a PostgreSQL/SQLite database, extracts features, performs model inference, stores the results, and can email the recommendation back to the user.

### Infrastructure & Deployment
The repository includes a complete production-ready setup containerized with Docker compose.production.yaml and compose.production.rds.yaml. It is integrated with a GitHub Actions CI/CD pipeline in ci.yaml that runs automated pytest suites against a PostgreSQL instance, builds multi-architecture Docker images, pushes them to Amazon ECR Public, and deploys directly to AWS EC2 backed by AWS RDS.
