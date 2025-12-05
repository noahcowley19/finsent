import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
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
        warning: {
          DEFAULT: '#f59e0b',
          light: '#fef3c7',
          dark: '#92400e',
        },
        primary: '#1e293b',
        secondary: '#64748b',
        tertiary: '#94a3b8',
        background: '#f8fafc',
        'card-bg': '#ffffff',
        border: '#e2e8f0',
        'border-light': '#f1f5f9',
      },
      boxShadow: {
        'custom': '0 1px 3px rgba(15, 23, 42, 0.08)',
        'custom-hover': '0 8px 30px rgba(15, 23, 42, 0.12)',
        'card': '0 1px 3px rgba(15, 23, 42, 0.08)',
        'card-hover': '0 8px 30px rgba(15, 23, 42, 0.12)',
        'btn': '0 4px 12px rgba(15, 23, 42, 0.12)',
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
        '3xl': '20px',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'spin-slow': 'spin 1.5s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
