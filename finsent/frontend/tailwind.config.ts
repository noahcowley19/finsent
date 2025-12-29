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
      // COLORS - Editorial Finance Palette
      // ========================================================================
      colors: {
        // Cream - Main background
        cream: {
          DEFAULT: "#FDFBF7",
          50: "#FDFBF7",
          100: "#FAF8F3",
          200: "#F5F2EA",
          300: "#EDE9DD",
          400: "#DED8C8",
        },

        // Ink Black - Primary text and headings
        "ink-black": {
          DEFAULT: "#0A0A0A",
          900: "#0A0A0A",
          800: "#1A1A1A",
          700: "#2A2A2A",
        },

        // Navy - Secondary text, body copy, active states
        navy: {
          DEFAULT: "#1F2937",
          50: "#F9FAFB",
          100: "#F3F4F6",
          200: "#E5E7EB",
          300: "#D1D5DB",
          400: "#9CA3AF",
          500: "#6B7280",
          600: "#4B5563",
          700: "#374151",
          800: "#1F2937",
          900: "#1B2A4E", // Deep Navy for navbar and active states
        },

        // Terracotta - Action color, highlights, CTAs
        terracotta: {
          DEFAULT: "#BC4B51",
          50: "#FEF2F2",
          100: "#FEE2E2",
          200: "#FECACA",
          300: "#FCA5A5",
          400: "#F87171",
          500: "#BC4B51", // Primary action color
          600: "#A53F44",
          700: "#8E3439",
          800: "#77292D",
          900: "#601E22",
        },

        // Borders
        border: {
          DEFAULT: "#E5E7EB", // Subtle
          medium: "#D1D5DB",  // Medium
        },

        // Success - Green for positive trends
        success: {
          50: "#F0FDF4",
          100: "#DCFCE7",
          200: "#BBF7D0",
          300: "#86EFAC",
          400: "#4ADE80",
          500: "#22C55E",
          600: "#16A34A",
          700: "#15803D",
        },

        // Error - Red for warnings
        error: {
          50: "#FEF2F2",
          100: "#FEE2E2",
          200: "#FECACA",
          300: "#FCA5A5",
          400: "#F87171",
          500: "#EF4444",
          600: "#DC2626",
          700: "#B91C1C",
        },

        // Warning - Amber
        warning: {
          50: "#FFFBEB",
          100: "#FEF3C7",
          200: "#FDE68A",
          300: "#FCD34D",
          400: "#FBBF24",
          500: "#F59E0B",
          600: "#D97706",
          700: "#B45309",
        },

        // Legacy aliases for backward compatibility
        obsidian: {
          950: "#0A0A0A",
          900: "#1A1A1D",
          850: "#222225",
          800: "#2C2C30",
          700: "#3D3D42",
          600: "#52525A",
          500: "#71717A",
          400: "#A1A1AA",
          300: "#D1D5DB",
          200: "#E4E4E7",
          100: "#F4F4F5",
        },

        ink: {
          950: "#0A0A0A",
          900: "#1A1A1D",
          850: "#222225",
          800: "#2C2C30",
          700: "#3D3D42",
          600: "#52525A",
          500: "#71717A",
          400: "#A1A1AA",
          300: "#D1D5DB",
          200: "#E4E4E7",
          100: "#F4F4F5",
          50: "#FDFCF9",
        },

        // Gray palette to match navy tones
        gray: {
          50: "#F9FAFB",
          100: "#F3F4F6",
          200: "#E5E7EB",
          300: "#D1D5DB",
          400: "#9CA3AF",
          500: "#6B7280",
          600: "#4B5563",
          700: "#374151",
          800: "#1F2937",
          900: "#111827",
        },

        // Electric Blue (kept for backward compatibility)
        electric: {
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
        },

        accent: {
          DEFAULT: "#BC4B51", // Terracotta
          50: "#FEF2F2",
          100: "#FEE2E2",
          200: "#FECACA",
          300: "#FCA5A5",
          400: "#F87171",
          500: "#BC4B51",
          600: "#A53F44",
          700: "#8E3439",
        },
      },

      // ========================================================================
      // TYPOGRAPHY - Clean Sans-Serif
      // ========================================================================
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        display: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },

      fontSize: {
        // Display sizes with tight tracking
        "display-2xl": ["4.5rem", { lineHeight: "1", letterSpacing: "-0.02em", fontWeight: "700" }],
        "display-xl": ["3.75rem", { lineHeight: "1", letterSpacing: "-0.02em", fontWeight: "700" }],
        "display-lg": ["3rem", { lineHeight: "1.1", letterSpacing: "-0.02em", fontWeight: "700" }],
        "display-md": ["2.25rem", { lineHeight: "1.15", letterSpacing: "-0.02em", fontWeight: "700" }],
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
        tightest: "-0.02em",
        tighter: "-0.015em",
        tight: "-0.01em",
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
      // BORDER RADIUS - Editorial Style
      // ========================================================================
      borderRadius: {
        sm: "0.375rem",
        md: "0.5rem",
        lg: "0.5rem",     // 8px for buttons
        xl: "0.75rem",    // 12px for cards
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
      },

      // ========================================================================
      // BOX SHADOWS - Subtle Editorial Shadows
      // ========================================================================
      boxShadow: {
        "xs": "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        "sm": "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
        "md": "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
        "lg": "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
        "xl": "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)",
        "2xl": "0 25px 50px -12px rgb(0 0 0 / 0.25)",
        "inner": "inset 0 2px 4px 0 rgb(0 0 0 / 0.05)",
        "none": "none",
      },

      // ========================================================================
      // ANIMATION
      // ========================================================================
      transitionDuration: {
        instant: "75ms",
        fast: "150ms",
        normal: "200ms",
        slow: "300ms",
        slower: "500ms",
        slowest: "700ms",
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
          "0%": { opacity: "0", transform: "translateY(20px)" },
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
      },

      animation: {
        "fade-in": "fade-in 200ms ease-out forwards",
        "fade-in-up": "fade-in-up 500ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in-down": "fade-in-down 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "scale-in": "scale-in 200ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-right": "slide-in-right 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-left": "slide-in-left 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-up": "slide-up 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "shimmer": "shimmer 2s linear infinite",
        "pulse-subtle": "pulse-subtle 2s ease-in-out infinite",
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
    },
  },
  plugins: [],
};

export default config;
