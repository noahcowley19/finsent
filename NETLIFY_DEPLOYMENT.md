# Netlify Deployment Guide

This guide will help you deploy the FinSent frontend to Netlify successfully on the first try.

## Prerequisites

Before deploying to Netlify, ensure you have:

1. ✅ A Netlify account (free tier works fine)
2. ✅ Your backend API deployed and accessible (e.g., on Render)
3. ✅ A PostgreSQL database (Supabase, Neon, or Railway recommended)
4. ✅ Generated a NEXTAUTH_SECRET value

## Step 1: Prepare Environment Variables

You'll need to configure these environment variables in Netlify. **Have them ready before you start the deployment.**

### Required Variables:

```bash
# Backend API URL - IMPORTANT: Use your actual backend URL
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com

# Database connection string
DATABASE_URL=postgresql://user:password@host:5432/database

# NextAuth secret (generate with: openssl rand -base64 32)
NEXTAUTH_SECRET=your-generated-secret-here

# Your Netlify site URL (will be provided after first deploy)
NEXTAUTH_URL=https://your-site.netlify.app
```

### How to Generate NEXTAUTH_SECRET:

On your local machine or any Unix-like system:
```bash
openssl rand -base64 32
```

Or use an online generator: https://generate-secret.vercel.app/32

## Step 2: Connect Repository to Netlify

1. Log in to [Netlify](https://app.netlify.com)
2. Click "Add new site" → "Import an existing project"
3. Choose "Deploy with GitHub"
4. Authorize Netlify to access your GitHub account
5. Select the `noahcowley19/finsent` repository

## Step 3: Configure Build Settings

On the "Site settings for finsent" page, configure:

### Basic build settings:

- **Base directory**: `finsent/frontend`
- **Build command**: `npm run build`
- **Publish directory**: `.next`

These are already configured in `netlify.toml`, so you can usually just click "Deploy site" and Netlify will read the config file.

## Step 4: Add Environment Variables

**CRITICAL STEP** - The build will fail without these!

1. Before or immediately after the first deploy, go to "Site settings" → "Environment variables"
2. Click "Add a variable" for each of the following:

```
NEXT_PUBLIC_API_URL = https://your-backend-url.onrender.com
DATABASE_URL = postgresql://user:password@host:5432/dbname
NEXTAUTH_SECRET = [your generated secret]
NEXTAUTH_URL = https://your-site.netlify.app
```

**Note**: For `NEXTAUTH_URL`, initially you can use a placeholder like `https://finsent.netlify.app`. After your first deploy, update it with your actual Netlify URL.

## Step 5: Update Backend CORS

Your backend needs to allow requests from your Netlify domain.

1. Go to your backend hosting service (e.g., Render)
2. Add an environment variable:
   ```
   ALLOWED_ORIGINS=http://localhost:3000,https://your-site.netlify.app,https://caveray.com
   ```
3. Redeploy your backend for the changes to take effect

## Step 6: Deploy!

1. Click "Deploy site" in Netlify
2. Wait for the build to complete (usually 2-5 minutes)
3. Once deployed, note your site URL (e.g., `https://xyz123.netlify.app`)

## Step 7: Post-Deployment Configuration

After your first successful deploy:

1. **Update NEXTAUTH_URL**:
   - Go to Site settings → Environment variables
   - Edit `NEXTAUTH_URL` to match your actual Netlify URL
   - Trigger a redeploy

2. **Set up custom domain** (optional):
   - Go to Site settings → Domain management
   - Add your custom domain (e.g., `caveray.com`)
   - Update `NEXTAUTH_URL` and backend `ALLOWED_ORIGINS` accordingly

3. **Test the site**:
   - Visit your Netlify URL
   - Try logging in (if you have seeded users)
   - Test a stock search to verify backend connectivity

## Troubleshooting

### Build Fails with "Failed to fetch fonts"

✅ **Already Fixed!** Fonts now load at runtime, not build time.

### Build Fails with "DATABASE_URL is not set"

- Double-check that you added `DATABASE_URL` to Netlify environment variables
- Make sure there are no typos in the variable name
- The connection string format should be: `postgresql://user:password@host:5432/database`

### Build Succeeds but Site Shows CORS Errors

- Your backend's `ALLOWED_ORIGINS` doesn't include your Netlify URL
- Update the backend environment variable to include your Netlify domain
- Redeploy the backend

### Build Fails with "Prisma Client not generated"

- This shouldn't happen with our updated `package.json` and `prisma-generate.sh` script
- If it does, check that the postinstall script is running correctly
- Verify `DATABASE_URL` is set

### API Calls Return 404

- Check that `NEXT_PUBLIC_API_URL` points to your actual backend URL
- Verify the backend is running and accessible
- Check browser console for the exact URL being called

## Verification Checklist

After deployment, verify these work:

- [ ] Site loads without errors
- [ ] Fonts display correctly
- [ ] Homepage renders properly
- [ ] Navigation menu works
- [ ] Stock search connects to backend
- [ ] No CORS errors in browser console
- [ ] Authentication flow works (if database is set up)

## Need Help?

If you encounter issues:

1. Check the **Netlify deploy logs** for build errors
2. Check the **browser console** for runtime errors
3. Verify all environment variables are set correctly
4. Ensure your backend is accessible and CORS is configured properly
5. Review the main `README.md` for additional troubleshooting tips

## Next Steps

Once deployed successfully:

1. Set up your database schema:
   ```bash
   DATABASE_URL=your-production-url npx prisma db push
   ```

2. Test all major features:
   - Stock search
   - Sentiment analysis
   - Financial analysis
   - Portfolio analysis

3. Monitor performance in Netlify Analytics

4. Set up custom domain if desired

---

**Deployment Status**: Ready for Netlify ✅

All configuration files are in place, dependencies are properly managed, and the build process has been tested successfully.
