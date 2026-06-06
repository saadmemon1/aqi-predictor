# Cron Job Automation Setup

Since we are hosting the system on **Hugging Face Spaces**, the space will automatically go to sleep after 48 hours of inactivity. To prevent this, and to automatically trigger our Feature Pipeline, we use [cron-job.org](https://cron-job.org).

## Steps to configure

1. Create a free account at [cron-job.org](https://cron-job.org).
2. Click on **Create Cronjob**.
3. **URL**: Enter the URL of your Hugging Face Space FastAPI endpoint.
   - E.g., `https://your-username-your-space-name.hf.space/run-feature-pipeline`
4. **Execution schedule**: Set it to run **every hour** (e.g. at minute 0).
5. **HTTP Method**: Change the method to `POST`.
6. Save the Cronjob.

This single cronjob will ping the `/run-feature-pipeline` endpoint every hour. This accomplishes two things:
1. It executes `feature_pipeline.py` to fetch the latest weather from OpenMeteo and uploads it to Hopsworks.
2. The incoming HTTP request keeps the Hugging Face Space awake 24/7 so your Streamlit dashboard is always accessible.
