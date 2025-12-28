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
      // COLORS - Linear-Modernist "Atmospheric Glass" Palette
      // ========================================================================
      colors: {
        // Cream - Warm light background (replaces cold white)
        cream: {
          50: "#FDFCF9",   // Primary background
          100: "#FAF8F3",  // Slightly darker
          200: "#F5F2EA",  // Card backgrounds
          300: "#EDE9DD",  // Borders
          400: "#DED8C8",  // Subtle elements
        },

        // Obsidian - Deep dark for text and buttons (replaces pure black)
        obsidian: {
          950: "#0F0F10",  // Darkest
          900: "#1A1A1D",  // Primary buttons
          850: "#222225",  // Button hover
          800: "#2C2C30",  // Secondary dark
          700: "#3D3D42",  // Muted dark
          600: "#52525A",  // Secondary text
          500: "#71717A",  // Muted text
          400: "#A1A1AA",  // Placeholder
          300: "#D4D4D8",  // Light borders
          200: "#E4E4E7",  // Very light
          100: "#F4F4F5",  // Near white
        },

        // Coral - Vibrant accent for growth indicators
        coral: {
          50: "#FFF5F5",
          100: "#FFE8E8",
          200: "#FFD0D0",
          300: "#FFADAD",
          400: "#FF7A7A",
          500: "#FF5C5C",  // Primary
          600: "#E53E3E",
          700: "#C53030",
        },

        // Electric Blue - Security icons and trust indicators
        electric: {
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",  // Primary
          600: "#2563EB",
          700: "#1D4ED8",
        },

        // Amber - Automation and AI features
        amber: {
          50: "#FFFBEB",
          100: "#FEF3C7",
          200: "#FDE68A",
          300: "#FCD34D",
          400: "#FBBF24",
          500: "#F59E0B",  // Primary
          600: "#D97706",
          700: "#B45309",
        },

        // Success - Green
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

        // Light Leak Colors
        lightLeak: {
          coral: "rgba(255, 180, 171, 0.15)",
          lavender: "rgba(196, 181, 253, 0.12)",
          peach: "rgba(255, 218, 185, 0.1)",
        },

        // Glass effects
        glass: {
          white: "rgba(255, 255, 255, 0.8)",
          whiteBorder: "rgba(255, 255, 255, 0.5)",
          dark: "rgba(0, 0, 0, 0.05)",
        },

        // Legacy ink colors (for backward compatibility)
        ink: {
          950: "#0F0F10",
          900: "#1A1A1D",
          850: "#222225",
          800: "#2C2C30",
          700: "#3D3D42",
          600: "#52525A",
          500: "#71717A",
          400: "#A1A1AA",
          300: "#D4D4D8",
          200: "#E4E4E7",
          100: "#F4F4F5",
          50: "#FDFCF9",
        },

        // Accent (alias to electric blue)
        accent: {
          DEFAULT: "#3B82F6",
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
          800: "#1E40AF",
          900: "#1E3A8A",
        },

        // Error
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

        // Warning (alias to amber)
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
      },

      // ========================================================================
      // TYPOGRAPHY - Grotesque Sans-Serif with tight tracking
      // ========================================================================
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        display: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },

      fontSize: {
        // Display sizes with -2% tracking for bold headlines
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
        tightest: "-0.02em",  // For bold display headlines
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
      // BORDER RADIUS - Modern rounded feel (32px for product windows)
      // ========================================================================
      borderRadius: {
        sm: "0.375rem",
        md: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
        "2xl": "1.25rem",
        "3xl": "1.5rem",
        "4xl": "2rem",       // 32px for floating product windows
      },

      // ========================================================================
      // BOX SHADOWS - Atmospheric diffuse shadows
      // ========================================================================
      boxShadow: {
        "xs": "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        "sm": "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
        "md": "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
        "lg": "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
        "xl": "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)",
        "2xl": "0 25px 50px -12px rgb(0 0 0 / 0.25)",
        // Atmospheric diffuse shadow for floating product windows
        "diffuse": "0 50px 100px -20px rgba(0, 0, 0, 0.1)",
        "diffuse-lg": "0 60px 120px -30px rgba(0, 0, 0, 0.12)",
        // Glass shadows
        "glass": "0 8px 32px 0 rgba(0, 0, 0, 0.08)",
        "glass-lg": "0 16px 48px 0 rgba(0, 0, 0, 0.1)",
        "glass-xl": "0 24px 64px 0 rgba(0, 0, 0, 0.12)",
        // Glow effects
        "glow-sm": "0 0 15px -3px rgb(59 130 246 / 0.3)",
        "glow-md": "0 0 25px -5px rgb(59 130 246 / 0.4)",
        "glow-lg": "0 0 35px -5px rgb(59 130 246 / 0.5)",
        "glow-coral": "0 0 25px -5px rgba(255, 92, 92, 0.3)",
        "inner": "inset 0 2px 4px 0 rgb(0 0 0 / 0.05)",
        "none": "none",
      },

      // ========================================================================
      // ANIMATION - Micro-interactions & scroll reveals
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
        // Chart drawing easing
        "chart-draw": "cubic-bezier(0.4, 0, 0.2, 1)",
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
        "float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
        "glow": {
          "0%, 100%": { boxShadow: "0 0 20px 0 rgba(59, 130, 246, 0.2)" },
          "50%": { boxShadow: "0 0 40px 5px rgba(59, 130, 246, 0.4)" },
        },
        // Chart line drawing animation
        "draw-line": {
          "0%": { strokeDashoffset: "1000" },
          "100%": { strokeDashoffset: "0" },
        },
        // Light leak pulse
        "light-pulse": {
          "0%, 100%": { opacity: "0.8" },
          "50%": { opacity: "1" },
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
        "float": "float 4s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite",
        "draw-line": "draw-line 1.5s cubic-bezier(0.4, 0, 0.2, 1) forwards",
        "light-pulse": "light-pulse 8s ease-in-out infinite",
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
        lg: "12px",  // Primary for glass cards
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
        // Light leak gradients
        "light-leak-coral": "radial-gradient(ellipse at top left, rgba(255, 180, 171, 0.25) 0%, transparent 50%)",
        "light-leak-lavender": "radial-gradient(ellipse at bottom right, rgba(196, 181, 253, 0.2) 0%, transparent 50%)",
        "light-leak-peach": "radial-gradient(ellipse at top right, rgba(255, 218, 185, 0.15) 0%, transparent 50%)",
        "noise": "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
};

export default config;
