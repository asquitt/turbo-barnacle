# Week 9: MLOps & Deployment ☁️

> **Deploy your system to production with complete MLOps stack**

## 🎯 Overview

Transform your working prototype into a production-ready system with experiment tracking, data versioning, monitoring, and cloud deployment.

## 📋 Learning Objectives

- ✅ Set up MLflow experiment tracking
- ✅ Implement DVC data versioning
- ✅ Integrate Weights & Biases
- ✅ Containerize with Docker
- ✅ Deploy to Kubernetes
- ✅ Set up monitoring and logging

## 🗓️ Time Estimate

12-15 hours over 7 days

## 🛠️ MLOps Stack

### 1. Experiment Tracking (MLflow)

Track every training run:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.001
    })

    # Train model
    for epoch in range(num_epochs):
        loss = train_epoch()
        mlflow.log_metric("train_loss", loss, step=epoch)

    mlflow.log_artifact("model.pth")
```

**Benefits:**
- Compare experiments
- Reproduce results
- Share with team

### 2. Data Versioning (DVC)

Version large datasets:
```bash
# Track data with DVC
dvc add data/materials_1000.pkl
git add data/materials_1000.pkl.dvc

# Push to remote storage
dvc push

# Pull latest data
dvc pull
```

**Benefits:**
- Version datasets like code
- Share data efficiently
- Reproduce experiments

### 3. Metrics Tracking (W&B)

Beautiful dashboards:
```python
import wandb

wandb.init(project="materials-discovery")
wandb.config.update({"lr": 0.001, "batch_size": 32})

for epoch in range(100):
    wandb.log({"loss": loss, "mae": mae})

wandb.log({"predictions": wandb.Table(data=df)})
```

**Benefits:**
- Real-time dashboards
- Hyperparameter sweeps
- Collaboration

### 4. Containerization (Docker)

Package entire system:
```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]
```

**Benefits:**
- Reproducible environments
- Easy deployment
- Isolation

### 5. Orchestration (Kubernetes)

Scale to production:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: materials-discovery
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: discovery
        image: materials-discovery:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
```

**Benefits:**
- Auto-scaling
- High availability
- Resource management

## 🛠️ Exercises

1. **MLflow Setup** (3 hours)
   - Install MLflow
   - Track experiments
   - Compare runs
   - Serve models

2. **DVC Integration** (2 hours)
   - Initialize DVC
   - Track datasets
   - Set up remote storage
   - Version data pipeline

3. **W&B Dashboards** (2 hours)
   - Create project
   - Log metrics
   - Build dashboards
   - Run hyperparameter sweeps

4. **Docker Deployment** (3 hours)
   - Write Dockerfile
   - Build image
   - Test container
   - Push to registry

5. **Kubernetes Basics** (2 hours)
   - Write deployment configs
   - Deploy to local cluster (minikube)
   - Test scaling
   - Set up monitoring

## ✅ Success Criteria

- [ ] MLflow tracking all experiments
- [ ] DVC versioning datasets
- [ ] W&B dashboards showing metrics
- [ ] Docker container runs successfully
- [ ] Kubernetes deployment working
- [ ] Monitoring alerts configured
- [ ] CI/CD pipeline set up (bonus)

## 📊 Production Checklist

### Before Deploying:
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Error handling robust
- [ ] Logging comprehensive
- [ ] Secrets managed securely
- [ ] Resource limits set
- [ ] Monitoring configured
- [ ] Backup strategy defined

### Deployment Steps:
1. Build Docker image
2. Push to container registry
3. Deploy to Kubernetes
4. Verify health checks
5. Monitor for issues
6. Set up alerts

## 🔒 Security Best Practices

```python
# ❌ Bad: Hardcoded secrets
api_key = "sk-1234567890"

# ✓ Good: Environment variables
import os
api_key = os.getenv("MP_API_KEY")

# ✓ Better: Secret management (K8s secrets, AWS Secrets Manager)
from kubernetes import client
api_key = get_secret("materials-discovery", "mp-api-key")
```

## 📈 Monitoring & Alerts

Track key metrics:
- **Model performance:** MAE, AUROC
- **System health:** CPU, memory, GPU usage
- **Discovery rate:** Materials per hour
- **Cost:** $ per discovery
- **Errors:** Failed predictions, API errors

Set up alerts:
- MAE > 0.15 eV → Retrain model
- Discovery rate < 10/hr → Investigation needed
- Cost > $0.15/discovery → Optimization needed

## 💰 Cost Optimization

Production costs:
- **Compute:** $50-200/month (depends on scale)
- **Storage:** $5-20/month
- **API calls:** $10-50/month
- **Total:** ~$100-300/month for small-scale deployment

Optimization strategies:
- Use spot instances (70% savings)
- Cache aggressively
- Batch requests
- Auto-scale based on demand

## 📚 Resources

**Tools:**
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [DVC Docs](https://dvc.org/doc)
- [W&B Docs](https://docs.wandb.ai/)
- [Docker Docs](https://docs.docker.com/)
- [Kubernetes Docs](https://kubernetes.io/docs/)

**Courses:**
- "Full Stack Deep Learning" - MLOps best practices
- "Made With ML" - Production ML systems

## 🚀 Getting Started

```bash
cd week-9-mlops

# Install tools
pip install mlflow dvc wandb

# Initialize services
mlflow ui  # localhost:5000
dvc init
wandb login

# Start exercises
cat exercises/exercise_1_mlflow.md
```

## 💡 Pro Tips

- **Start local:** Test on your machine before cloud
- **Version everything:** Code, data, models, configs
- **Monitor early:** Set up monitoring from day 1
- **Automate testing:** CI/CD catches issues early
- **Document deployment:** Write runbooks for common issues
- **Plan for failures:** Systems will fail, be prepared

## 🎓 Final Project

Deploy complete system to cloud:

1. **AWS/GCP/Azure setup**
2. **Container registry**
3. **Kubernetes cluster (EKS/GKE/AKS)**
4. **MLflow tracking server**
5. **DVC remote storage**
6. **Monitoring dashboards**
7. **CI/CD pipeline**
8. **Documentation site**

## 🎉 Congratulations!

You've completed all 9 weeks! You now have:

✅ Production-ready materials discovery system
✅ Complete MLOps infrastructure
✅ Deployed to cloud
✅ Monitoring and alerting
✅ Reproducible experiments
✅ Scalable architecture

**What's Next?**
- [ ] Apply to your research domain
- [ ] Publish results
- [ ] Contribute to open source
- [ ] Help others learn
- [ ] Build extensions
- [ ] Share your success!

---

**You did it!** 🚀🎉🏆

**Share your journey:**
- Blog post about what you learned
- GitHub repo with your code
- LinkedIn post celebrating completion
- Help the next learner!
