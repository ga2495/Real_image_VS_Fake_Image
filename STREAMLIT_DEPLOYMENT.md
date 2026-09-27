# Streamlit Community Cloud Deployment

## Required Python version

This project uses **Python 3.11** because the bundled FDCS-Net V4 model is deployed with
TensorFlow CPU 2.16.1.

The repository includes `.python-version` and `runtime.txt` documenting Python 3.11,
but **Streamlit Community Cloud's deployed Python version is selected in the deployment
Advanced settings**.

### Important if an existing deployment is using Python 3.14

If the Streamlit app was already deployed with Python 3.14, changing repository files
does not change the Python runtime of that existing app.

Use this procedure:

1. Open the app in Streamlit Community Cloud.
2. Delete the existing app.
3. Create/redeploy the app from the GitHub repository.
4. Open **Advanced settings**.
5. Select **Python 3.11**.
6. Set the main file to:
   `app.py`
7. Deploy.

Streamlit Community Cloud requires deleting and redeploying an existing app to change
its Python version.

## Repository

Recommended GitHub structure:

```text
Real_Fake_image_detetctor/
├── app.py
├── requirements.txt
├── requirements-training.txt
├── models/
│   └── fdcsnet_v4_final.keras
├── samples/
├── src/
├── configs/
├── notebooks/
└── reports/
```

The 18.9 MB model is intentionally tracked in Git and loaded locally by default.
