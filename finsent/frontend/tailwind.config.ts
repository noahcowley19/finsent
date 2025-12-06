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
        sans: ['Plus Jakarta Sans', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        // Core palette
        bg: {
          primary: '#06080d',
          secondary: '#0a0e17',
          tertiary: '#0f1420',
          card: '#111827',
          'card-hover': '#1a2236',
          elevated: '#151c2c',
        },
        // Accent
        accent: {
          DEFAULT: '#00d4aa',
          light: '#00f5c4',
          dark: '#00a88a',
        },
        // Semantic colors
        positive: {
          DEFAULT: '#00e5a0',
          light: 'rgba(0, 229, 160, 0.15)',
        },
        negative: {
          DEFAULT: '#ff6b6b',
          light: 'rgba(255, 107, 107, 0.15)',
        },
        warning: {
          DEFAULT: '#fbbf24',
          light: 'rgba(251, 191, 36, 0.15)',
        },
        neutral: {
          DEFAULT: '#64748b',
          light: 'rgba(100, 116, 139, 0.15)',
        },
        // Text hierarchy
        'text-primary': '#f8fafc',
        'text-secondary': '#94a3b8',
        'text-tertiary': '#64748b',
        'text-muted': '#475569',
        // Border
        border: {
          DEFAULT: 'rgba(255, 255, 255, 0.06)',
          light: 'rgba(255, 255, 255, 0.03)',
          accent: 'rgba(0, 212, 170, 0.3)',
        },
      },
      boxShadow: {
        'sm': '0 2px 8px rgba(0, 0, 0, 0.3)',
        'md': '0 4px 20px rgba(0, 0, 0, 0.4)',
        'lg': '0 8px 40px rgba(0, 0, 0, 0.5)',
        'glow': '0 0 40px rgba(0, 212, 170, 0.4)',
        'glow-sm': '0 0 20px rgba(0, 212, 170, 0.3)',
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
        '3xl': '20px',
        '4xl': '24px',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in-up': 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-glow': 'glow 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(0, 212, 170, 0.4)' },
          '50%': { boxShadow: '0 0 40px rgba(0, 212, 170, 0.4), 0 0 60px rgba(0, 212, 170, 0.4)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      transitionTimingFunction: {
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'out-back': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
        'in-out-circ': 'cubic-bezier(0.85, 0, 0.15, 1)',
      },
    },
  },
  plugins: [],
}

export default config
