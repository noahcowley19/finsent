# FinSent - Financial Sentiment Analysis Platform

A full-stack financial intelligence platform with AI-powered sentiment analysis, insider trading tracking, and quantitative analysis.

## Architecture

- **Frontend**: Next.js 14 (React) with TypeScript, deployed on Netlify
- **Backend**: Flask (Python) API, deployed on Render
- **Database**: PostgreSQL (for user auth and data storage)
- **Authentication**: NextAuth.js with Prisma adapter

## Prerequisites

- Node.js 20+ and npm
- Python 3.9+
- PostgreSQL database
- Netlify account (for frontend deployment)
- Render account (for backend deployment) or similar platform

## Local Development Setup

### Backend Setup

1. Navigate to the Backend directory:
   ```bash
   cd finsent/Backend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set environment variables (or create .env file):
   ```bash
   export PORT=5000
   export FLASK_ENV=development
   export ALLOWED_ORIGINS=http://localhost:3000
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd finsent/frontend
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env.local
   ```

3. Edit `.env.local` and configure:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:5000
   DATABASE_URL=postgresql://user:password@localhost:5432/finsent
   NEXTAUTH_SECRET=your-secret-here
   NEXTAUTH_URL=http://localhost:3000
   ```

4. Install dependencies:
   ```bash
   npm install
   ```

5. Run database migrations:
   ```bash
   npx prisma db push
   npx prisma generate
   ```

6. Start the development server:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

## Production Deployment

### Backend Deployment (Render)

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - **Root Directory**: `finsent/Backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 600`
4. Add environment variables:
   ```
   FLASK_ENV=production
   ALLOWED_ORIGINS=https://your-netlify-site.netlify.app,https://caveray.com
   RATE_LIMIT=15 per minute
   ```
5. Deploy!

### Frontend Deployment (Netlify)

1. Connect your GitHub repository to Netlify
2. Configure build settings:
   - **Base directory**: `finsent/frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `.next`
3. Add environment variables in Netlify UI:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   DATABASE_URL=your-postgresql-connection-string
   NEXTAUTH_SECRET=generate-with-openssl-rand-base64-32
   NEXTAUTH_URL=https://your-site.netlify.app
   ```
4. Enable the Next.js plugin if not already enabled
5. Deploy!

### Database Setup

1. Create a PostgreSQL database (recommended: Supabase, Neon, or Railway)
2. Get the connection string (DATABASE_URL)
3. Add it to your Netlify environment variables
4. Run migrations from your local machine:
   ```bash
   DATABASE_URL=your-production-url npx prisma db push
   ```

## Environment Variables Reference

### Frontend (.env.local)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://finsent-backend.onrender.com` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@host:5432/db` |
| `NEXTAUTH_SECRET` | NextAuth.js secret | Generate with `openssl rand -base64 32` |
| `NEXTAUTH_URL` | Your site URL | `https://your-site.netlify.app` |

### Backend (Environment Variables on Render)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | `5000` | No (set by host) |
| `FLASK_ENV` | Environment | `production` | Yes |
| `ALLOWED_ORIGINS` | CORS origins | See .env.example | Yes |
| `RATE_LIMIT` | API rate limit | `15 per minute` | No |

## API Endpoints

The backend provides the following API endpoints:

- `/api/analyze` - Sentiment analysis
- `/api/social-screening` - Social sentiment screening
- `/api/financials` - Financial analysis (Piotroski, Altman, Beneish)
- `/api/insider` - Insider trading analysis
- `/api/search` - Stock search and data
- `/api/portfolio/analyze` - Portfolio analysis
- `/api/quant-lab` - Quantitative analysis

See `frontend/lib/api.ts` for complete API documentation.

## Troubleshooting

### Build Fails with Font Errors
- **Issue**: Google Fonts cannot be fetched during build
- **Solution**: Fonts are now loaded at runtime via HTML link tags in the layout

### CORS Errors
- **Issue**: Frontend can't connect to backend
- **Solution**: Add your Netlify URL to `ALLOWED_ORIGINS` in backend environment variables

### Database Connection Errors
- **Issue**: Prisma can't connect during build
- **Solution**: Ensure `DATABASE_URL` is set in Netlify environment variables

### Node Modules in Git
- **Issue**: Large files in git
- **Solution**: Already fixed - `node_modules/` is now properly gitignored

## Security Notes

- Never commit `.env` or `.env.local` files
- Keep `NEXTAUTH_SECRET` and database credentials secure
- Use environment variables for all sensitive data
- The backend includes bot/scraper protection and rate limiting

## Support

For issues or questions, please open an issue on GitHub.
