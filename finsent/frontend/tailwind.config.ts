import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    // ==========================================================================
    // CUSTOM CONTAINER
    // ==========================================================================
    container: {
      center: true,
      padding: {
        DEFAULT: "1rem",
        sm: "1.5rem",
        lg: "2rem",
      },
      screens: {
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
        "2xl": "1440px",
      },
    },

    extend: {
      // ========================================================================
      // COLORS - Modern SaaS Palette (Linear/Apple inspired)
      // ========================================================================
      colors: {
        // Deep Ink - Primary dark (replaces pure black)
        ink: {
          950: "#020617", // Primary background
          900: "#0f172a", // Slightly lighter
          850: "#131B2E", // Card backgrounds
          800: "#1e293b", // Elevated surfaces
          700: "#334155", // Borders, muted elements
          600: "#475569", // Secondary text
          500: "#64748b", // Muted text
          400: "#94a3b8", // Placeholder text
          300: "#cbd5e1", // Light borders
          200: "#e2e8f0", // Very light backgrounds
          100: "#f1f5f9", // Near white
          50: "#f8fafc",  // White-ish
        },

        // Accent - Subtle blue-gray (can be customized)
        accent: {
          DEFAULT: "#3b82f6",
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
        },

        // Success
        success: {
          50: "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
        },

        // Warning
        warning: {
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
          700: "#b45309",
        },

        // Error
        error: {
          50: "#fef2f2",
          100: "#fee2e2",
          200: "#fecaca",
          300: "#fca5a5",
          400: "#f87171",
          500: "#ef4444",
          600: "#dc2626",
          700: "#b91c1c",
        },

        // Glass borders
        glass: {
          light: "rgba(255, 255, 255, 0.1)",
          medium: "rgba(255, 255, 255, 0.15)",
          heavy: "rgba(255, 255, 255, 0.2)",
          dark: "rgba(0, 0, 0, 0.1)",
        },
      },

      // ========================================================================
      // TYPOGRAPHY - Inter (Humanist Sans-Serif)
      // ========================================================================
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        display: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },

      fontSize: {
        "display-2xl": ["4.5rem", { lineHeight: "1", letterSpacing: "-0.025em", fontWeight: "600" }],
        "display-xl": ["3.75rem", { lineHeight: "1", letterSpacing: "-0.025em", fontWeight: "600" }],
        "display-lg": ["3rem", { lineHeight: "1.1", letterSpacing: "-0.025em", fontWeight: "600" }],
        "display-md": ["2.25rem", { lineHeight: "1.15", letterSpacing: "-0.02em", fontWeight: "600" }],
        "display-sm": ["1.875rem", { lineHeight: "1.2", letterSpacing: "-0.02em", fontWeight: "600" }],
        "heading-xl": ["1.5rem", { lineHeight: "1.25", letterSpacing: "-0.015em", fontWeight: "600" }],
        "heading-lg": ["1.25rem", { lineHeight: "1.3", letterSpacing: "-0.01em", fontWeight: "600" }],
        "heading-md": ["1.125rem", { lineHeight: "1.4", letterSpacing: "-0.01em", fontWeight: "500" }],
        "heading-sm": ["1rem", { lineHeight: "1.5", letterSpacing: "-0.01em", fontWeight: "500" }],
        "body-lg": ["1.125rem", { lineHeight: "1.6" }],
        "body-md": ["1rem", { lineHeight: "1.6" }],
        "body-sm": ["0.875rem", { lineHeight: "1.5" }],
        "body-xs": ["0.75rem", { lineHeight: "1.4" }],
        caption: ["0.75rem", { lineHeight: "1.4" }],
        overline: ["0.75rem", { lineHeight: "1.4", letterSpacing: "0.05em", fontWeight: "500" }],
      },

      letterSpacing: {
        tighter: "-0.025em",
        tight: "-0.015em",
        normal: "0",
        wide: "0.01em",
        wider: "0.025em",
        widest: "0.05em",
      },

      // ========================================================================
      // SPACING
      // ========================================================================
      spacing: {
        "4.5": "1.125rem",
        "13": "3.25rem",
        "15": "3.75rem",
        "18": "4.5rem",
        "22": "5.5rem",
        "26": "6.5rem",
        "30": "7.5rem",
        "nav": "4rem",
        "nav-scrolled": "3.5rem",
      },

      // ========================================================================
      // SIZING
      // ========================================================================
      maxWidth: {
        "8xl": "88rem",
        "9xl": "96rem",
      },

      height: {
        "input-sm": "2rem",
        "input-md": "2.5rem",
        "input-lg": "2.75rem",
        "input-xl": "3rem",
        "button-sm": "2rem",
        "button-md": "2.5rem",
        "button-lg": "2.75rem",
        "button-xl": "3rem",
        nav: "4rem",
        "nav-scrolled": "3.5rem",
      },

      // ========================================================================
      // BORDER RADIUS - More rounded for modern feel
      // ========================================================================
      borderRadius: {
        sm: "0.375rem",
        md: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
        "2xl": "1.25rem",
        "3xl": "1.5rem",
      },

      // ========================================================================
      // BOX SHADOWS - Softer, atmospheric
      // ========================================================================
      boxShadow: {
        "xs": "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        "sm": "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
        "md": "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
        "lg": "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
        "xl": "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)",
        "2xl": "0 25px 50px -12px rgb(0 0 0 / 0.25)",
        "glow-sm": "0 0 15px -3px rgb(59 130 246 / 0.3)",
        "glow-md": "0 0 25px -5px rgb(59 130 246 / 0.4)",
        "glow-lg": "0 0 35px -5px rgb(59 130 246 / 0.5)",
        "inner": "inset 0 2px 4px 0 rgb(0 0 0 / 0.05)",
        "glass": "0 8px 32px 0 rgba(0, 0, 0, 0.12)",
        "glass-lg": "0 16px 48px 0 rgba(0, 0, 0, 0.15)",
        "none": "none",
      },

      // ========================================================================
      // ANIMATION - Micro-interactions
      // ========================================================================
      transitionDuration: {
        instant: "75ms",
        fast: "150ms",
        normal: "200ms",
        slow: "300ms",
        slower: "500ms",
      },

      transitionTimingFunction: {
        "ease-out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
        "ease-out-back": "cubic-bezier(0.34, 1.56, 0.64, 1)",
        "spring": "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
        "bounce": "cubic-bezier(0.68, -0.55, 0.265, 1.55)",
      },

      keyframes: {
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in-down": {
          "0%": { opacity: "0", transform: "translateY(-10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        "slide-in-right": {
          "0%": { transform: "translateX(100%)" },
          "100%": { transform: "translateX(0)" },
        },
        "slide-in-left": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(0)" },
        },
        "slide-up": {
          "0%": { transform: "translateY(100%)" },
          "100%": { transform: "translateY(0)" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-subtle": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-5px)" },
        },
        "glow": {
          "0%, 100%": { boxShadow: "0 0 20px 0 rgba(59, 130, 246, 0.3)" },
          "50%": { boxShadow: "0 0 30px 5px rgba(59, 130, 246, 0.5)" },
        },
      },

      animation: {
        "fade-in": "fade-in 200ms ease-out forwards",
        "fade-in-up": "fade-in-up 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in-down": "fade-in-down 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "scale-in": "scale-in 200ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-right": "slide-in-right 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-left": "slide-in-left 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-up": "slide-up 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "shimmer": "shimmer 2s linear infinite",
        "pulse-subtle": "pulse-subtle 2s ease-in-out infinite",
        "float": "float 3s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite",
        "spin-slow": "spin 3s linear infinite",
      },

      // ========================================================================
      // Z-INDEX
      // ========================================================================
      zIndex: {
        below: "-1",
        base: "0",
        above: "1",
        dropdown: "10",
        sticky: "20",
        fixed: "30",
        "modal-backdrop": "40",
        modal: "50",
        popover: "60",
        tooltip: "70",
        toast: "80",
        max: "9999",
      },

      // ========================================================================
      // BACKDROP BLUR
      // ========================================================================
      backdropBlur: {
        xs: "2px",
        sm: "4px",
        md: "8px",
        lg: "12px",
        xl: "16px",
        "2xl": "24px",
        "3xl": "40px",
      },

      // ========================================================================
      // BACKGROUNDS
      // ========================================================================
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "gradient-subtle": "linear-gradient(to bottom right, var(--tw-gradient-stops))",
        "gradient-glow": "radial-gradient(ellipse at center, var(--tw-gradient-stops))",
        "noise": "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
};

export default config;
