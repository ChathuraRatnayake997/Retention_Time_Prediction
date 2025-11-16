# GitHub Pages Deployment Guide for HPLC Research Website

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner → "New repository"
3. Repository name suggestions:
   - `hplc-retention-time-prediction`
   - `hplc-research-website`
   - `chromatography-ml`
4. Make it **Public** (GitHub Pages requires public repositories for free accounts)
5. Check "Add a README file"
6. Click "Create repository"

## Step 2: Prepare Your Files

Your repository should contain these files:
```
hplc-research-website/
├── index.html          (your main website file)
├── data/
│   └── plots/
│       └── simplified_model_analysis.png
├── main.py
├── src/
├── requirements.txt
└── README.md
```

## Step 3: Upload Your Files

### Option A: Using GitHub Web Interface (Recommended for beginners)

1. **Upload index.html**:
   - Click "uploading an existing file" link on your repository
   - Drag and drop your `index.html` file
   - Commit directly to main branch

2. **Upload the plot**:
   - Create a `data/plots/` folder structure
   - Upload `simplified_model_analysis.png` to `data/plots/`

3. **Upload your Python project** (optional but recommended):
   - Upload your `main.py`, `src/` folder, `requirements.txt`
   - Add a proper `README.md` explaining your research

### Option B: Using Git Commands (If you have Git installed)

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Copy your files to the repository folder
# Copy index.html, data/, src/, main.py, requirements.txt

# Add all files
git add .

# Commit
git commit -m "Initial commit: HPLC retention time prediction research website"

# Push to GitHub
git push origin main
```

## Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on "Settings" tab (top of repository)
3. Scroll down to "Pages" section in the left sidebar
4. Under "Source", select "Deploy from a branch"
5. Select "main" branch and "/ (root)" folder
6. Click "Save"

## Step 5: Access Your Website

After 1-2 minutes, your website will be available at:
`https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

## Step 6: Custom Domain (Optional)

If you want a custom domain like `hplc-research.com`:

1. Add a `CNAME` file to your repository root with your domain name
2. Configure your domain's DNS settings
3. Update GitHub Pages settings

## Troubleshooting

### Common Issues:

1. **Plot not showing**: 
   - Ensure the image path is correct: `data/plots/simplified_model_analysis.png`
   - Check that the file exists in your repository

2. **Website not loading**:
   - Wait 5-10 minutes for GitHub Pages to deploy
   - Check the "Pages" section in repository settings for error messages

3. **Styling issues**:
   - Ensure `index.html` is in the root directory
   - Check browser console for any errors

## Recommended README.md Content

Create a comprehensive README.md for your repository:

```markdown
# HPLC Retention Time Prediction

Machine learning approach to predict HPLC retention times for pharmaceutical compounds.

## Results

- **Model**: Ridge Regression
- **Performance**: RMSE: 2.262, R²: 0.633
- **Dataset**: 114 pharmaceutical compounds
- **Key Feature**: xlogp3 (lipophilicity coefficient)

## Live Demo

Visit the [research website](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/) to see detailed results and visualizations.

## Files

- `index.html` - Research website
- `data/plots/` - Model analysis visualizations
- `src/` - Complete ML pipeline code
- `main.py` - Main execution script

## Citation

If you use this research, please cite:
```
```

## Next Steps After Deployment

1. **Share your research**: Send the GitHub Pages URL to colleagues and collaborators
2. **Add documentation**: Include detailed methodology and results in your README
3. **SEO optimization**: Add meta descriptions and keywords for better discoverability
4. **Analytics**: Add Google Analytics to track visitors (optional)
5. **Blog integration**: Consider adding a blog section for research updates

## Contact

Update the contact section in `index.html` with your actual email and research group information.