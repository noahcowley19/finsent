import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        positive: {
          DEFAULT: '#10b981',
          light: '#d1fae5',
          dark: '#065f46',
        },
        negative: {
          DEFAULT: '#ef4444',
          light: '#fee2e2',
          dark: '#991b1b',
        },
        neutral: {
          DEFAULT: '#6b7280',
          light: '#f3f4f6',
          dark: '#374151',
        },
        primary: '#1e293b',
        secondary: '#64748b',
        background: '#f8fafc',
        'card-bg': '#ffffff',
        border: '#e2e8f0',
      },
      boxShadow: {
        'custom': '0 1px 3px rgba(15, 23, 42, 0.08)',
        'custom-hover': '0 4px 12px rgba(15, 23, 42, 0.12)',
      }
    },
  },
  plugins: [],
}
export default config
