# Automated Pipeline Setup (GitHub Actions + cron-job.org)

To automatically fetch fresh weather data and push it to Hopsworks every hour, we use **GitHub Actions**. However, because GitHub's built-in hourly cron schedule is famously unreliable (often delayed by hours), we use [cron-job.org](https://cron-job.org) as an external, highly accurate trigger.

## Step 1: Generate a GitHub Personal Access Token (PAT)
For an external service to trigger your GitHub Action, it needs permission.
1. Go to your GitHub account settings -> Developer Settings -> Personal access tokens -> **Tokens (classic)**.
2. Click **Generate new token (classic)**.
3. Give it a note (e.g., "cron-job-trigger").
4. Select the **`repo`** scope (this gives it access to trigger workflows in your repositories).
5. Generate the token and **copy it** (you won't be able to see it again).

## Step 2: Configure cron-job.org
1. Create a free account at [cron-job.org](https://cron-job.org).
2. Click **Create Cronjob**.
3. **URL**: Enter `https://api.github.com/repos/YOUR-USERNAME/aqi-predictor/dispatches`
   *(Replace YOUR-USERNAME with your actual GitHub username).*
4. **Execution schedule**: Set it to run **every hour** (e.g., at minute 0).
5. Switch to the **Advanced** tab:
   - **HTTP Method**: Change to `POST`.
   - **Request Headers**: Add three headers:
     1. Key: `Accept`, Value: `application/vnd.github.v3+json`
     2. Key: `Authorization`, Value: `Bearer YOUR_COPIED_TOKEN`
     3. Key: `Content-Type`, Value: `application/json`
   - **Request Body**: Select `Raw` and paste:
     ```json
     {"event_type": "run-feature-pipeline"}
     ```
6. Save the Cronjob.

This setup ensures that precisely every hour, GitHub Actions spins up an Ubuntu server, installs your libraries, and executes `src/pipelines/feature_pipeline.py`. Your execution logs will be perfectly visible in the **Actions** tab of your repository!
