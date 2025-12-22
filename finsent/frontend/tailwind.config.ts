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
      // COLORS
      // ========================================================================
      colors: {
        // Primary: Cream (Backgrounds)
        cream: {
          50: "#FAF7F2",
          100: "#EFE4D2",
          200: "#E5D9C3",
          300: "#D4C4A8",
          400: "#C4B08D",
        },
        
        // Primary: Navy (Brand, Text)
        navy: {
          50: "#E8F1F8",
          100: "#B8D1E5",
          200: "#7BA3C2",
          300: "#4A7A9D",
          400: "#3A6285",
          500: "#254D70",
          600: "#2A3F6A",
          700: "#1E2E5A",
          800: "#1A2759",
          900: "#131D4F",
        },
        
        // Accent: Terracotta (CTAs)
        terra: {
          50: "#FBF3EF",
          100: "#F5E6DE",
          200: "#E8BBA8",
          300: "#D4917A",
          400: "#B86A4A",
          500: "#954C2E",
          600: "#7A3D24",
          700: "#6B3318",
        },
        
        // Semantic: Success
        success: {
          50: "#F0FAF4",
          100: "#E8F5ED",
          200: "#A3D9B8",
          300: "#6FBD8F",
          400: "#4A9D6F",
          500: "#2D7A4F",
          600: "#246B44",
          700: "#1D5A38",
        },
        
        // Semantic: Warning
        warning: {
          50: "#FFFBF0",
          100: "#FDF6E3",
          200: "#E8CDA0",
          300: "#D4A85C",
          400: "#B8893D",
          500: "#9A6B28",
          600: "#8A5F1F",
          700: "#7A5319",
        },
        
        // Semantic: Error
        error: {
          50: "#FEF5F5",
          100: "#FDEBEB",
          200: "#F5B3B3",
          300: "#E88080",
          400: "#D45A5A",
          500: "#B83A3A",
          600: "#A12F2F",
          700: "#8A2525",
        },
        
        // Border colors (using CSS custom properties)
        border: {
          light: "rgba(19, 29, 79, 0.08)",
          medium: "rgba(19, 29, 79, 0.12)",
          heavy: "rgba(19, 29, 79, 0.20)",
        },
      },
      
      // ========================================================================
      // TYPOGRAPHY
      // ========================================================================
      fontFamily: {
        display: ["Fraunces", "Georgia", "Times New Roman", "serif"],
        heading: ["Plus Jakarta Sans", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        body: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },
      
      fontSize: {
        "display-xl": ["4.5rem", { lineHeight: "1.1", letterSpacing: "-0.03em" }],
        "display-lg": ["3.5rem", { lineHeight: "1.15", letterSpacing: "-0.025em" }],
        "display-md": ["2.5rem", { lineHeight: "1.2", letterSpacing: "-0.02em" }],
        "display-sm": ["2rem", { lineHeight: "1.25", letterSpacing: "-0.02em" }],
        "heading-xl": ["1.75rem", { lineHeight: "1.25" }],
        "heading-lg": ["1.5rem", { lineHeight: "1.3" }],
        "heading-md": ["1.25rem", { lineHeight: "1.4" }],
        "heading-sm": ["1.125rem", { lineHeight: "1.5" }],
        "body-lg": ["1.125rem", { lineHeight: "1.6" }],
        "body-md": ["1rem", { lineHeight: "1.6" }],
        "body-sm": ["0.875rem", { lineHeight: "1.5" }],
        caption: ["0.75rem", { lineHeight: "1.4" }],
        overline: ["0.75rem", { lineHeight: "1.4", letterSpacing: "0.1em" }],
      },
      
      letterSpacing: {
        tighter: "-0.03em",
        tight: "-0.02em",
        normal: "0",
        wide: "0.01em",
        wider: "0.05em",
        widest: "0.1em",
      },
      
      // ========================================================================
      // SPACING (extending default Tailwind spacing)
      // ========================================================================
      spacing: {
        "4.5": "1.125rem",  // 18px
        "13": "3.25rem",    // 52px
        "15": "3.75rem",    // 60px
        "18": "4.5rem",     // 72px
        "22": "5.5rem",     // 88px
        "26": "6.5rem",     // 104px
        "30": "7.5rem",     // 120px
        "34": "8.5rem",     // 136px
        "38": "9.5rem",     // 152px
        "42": "10.5rem",    // 168px
        "46": "11.5rem",    // 184px
        "50": "12.5rem",    // 200px
        "54": "13.5rem",    // 216px
        "58": "14.5rem",    // 232px
        "62": "15.5rem",    // 248px
        "66": "16.5rem",    // 264px
        "70": "17.5rem",    // 280px
        "nav": "5rem",      // 80px - nav height
        "nav-scrolled": "4rem", // 64px - scrolled nav height
      },
      
      // ========================================================================
      // SIZING
      // ========================================================================
      maxWidth: {
        "container-xs": "20rem",    // 320px
        "container-sm": "24rem",    // 384px
        "container-md": "28rem",    // 448px
        "container-lg": "32rem",    // 512px
        "container-xl": "36rem",    // 576px
        "container-2xl": "42rem",   // 672px
        "container-3xl": "48rem",   // 768px
        "container-4xl": "56rem",   // 896px
        "container-5xl": "64rem",   // 1024px
        "container-6xl": "72rem",   // 1152px
        "container-7xl": "80rem",   // 1280px
        "container-max": "90rem",   // 1440px
      },
      
      height: {
        "input-sm": "2rem",      // 32px
        "input-md": "2.5rem",    // 40px
        "input-lg": "3rem",      // 48px
        "input-xl": "3.5rem",    // 56px
        "button-sm": "2rem",     // 32px
        "button-md": "2.5rem",   // 40px
        "button-lg": "3rem",     // 48px
        "button-xl": "3.5rem",   // 56px
        nav: "5rem",             // 80px
        "nav-scrolled": "4rem",  // 64px
      },
      
      // ========================================================================
      // BORDER RADIUS
      // ========================================================================
      borderRadius: {
        sm: "0.375rem",   // 6px
        md: "0.5rem",     // 8px
        lg: "0.75rem",    // 12px
        xl: "1rem",       // 16px
        "2xl": "1.5rem",  // 24px
        "3xl": "2rem",    // 32px
      },
      
      // ========================================================================
      // BOX SHADOWS
      // ========================================================================
      boxShadow: {
        xs: "0 1px 2px rgba(19, 29, 79, 0.04)",
        sm: "0 2px 4px rgba(19, 29, 79, 0.06)",
        md: "0 4px 12px rgba(19, 29, 79, 0.08)",
        lg: "0 8px 24px rgba(19, 29, 79, 0.10)",
        xl: "0 16px 48px rgba(19, 29, 79, 0.12)",
        "2xl": "0 24px 64px rgba(19, 29, 79, 0.16)",
        inner: "inset 0 2px 4px rgba(19, 29, 79, 0.04)",
        terra: "0 4px 14px rgba(149, 76, 46, 0.25)",
        "terra-lg": "0 8px 24px rgba(149, 76, 46, 0.30)",
        success: "0 4px 14px rgba(45, 122, 79, 0.25)",
        error: "0 4px 14px rgba(184, 58, 58, 0.25)",
        none: "0 0 0 0 transparent",
      },
      
      // ========================================================================
      // ANIMATION
      // ========================================================================
      transitionDuration: {
        instant: "100ms",
        fast: "200ms",
        normal: "300ms",
        slow: "400ms",
        slower: "600ms",
        slowest: "800ms",
      },
      
      transitionTimingFunction: {
        "ease-out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
        "ease-out-back": "cubic-bezier(0.34, 1.56, 0.64, 1)",
        spring: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
        bounce: "cubic-bezier(0.68, -0.55, 0.265, 1.55)",
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
          "0%": { opacity: "0", transform: "translateY(-20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        "slide-in-up": {
          "0%": { transform: "translateY(100%)" },
          "100%": { transform: "translateY(0)" },
        },
        "slide-in-right": {
          "0%": { transform: "translateX(100%)" },
          "100%": { transform: "translateX(0)" },
        },
        "slide-in-left": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        spin: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "10%, 30%, 50%, 70%, 90%": { transform: "translateX(-4px)" },
          "20%, 40%, 60%, 80%": { transform: "translateX(4px)" },
        },
        "modal-enter": {
          "0%": { opacity: "0", transform: "scale(0.95) translateY(10px)" },
          "100%": { opacity: "1", transform: "scale(1) translateY(0)" },
        },
        "modal-exit": {
          "0%": { opacity: "1", transform: "scale(1) translateY(0)" },
          "100%": { opacity: "0", transform: "scale(0.95) translateY(10px)" },
        },
        "toast-enter": {
          "0%": { opacity: "0", transform: "translateX(100%) scale(0.9)" },
          "100%": { opacity: "1", transform: "translateX(0) scale(1)" },
        },
        "toast-exit": {
          "0%": { opacity: "1", transform: "translateX(0) scale(1)" },
          "100%": { opacity: "0", transform: "translateX(100%) scale(0.9)" },
        },
        bounce: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-25%)" },
        },
        wiggle: {
          "0%, 100%": { transform: "rotate(0deg)" },
          "25%": { transform: "rotate(-3deg)" },
          "75%": { transform: "rotate(3deg)" },
        },
      },
      
      animation: {
        "fade-in": "fade-in 300ms ease-out forwards",
        "fade-in-up": "fade-in-up 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in-down": "fade-in-down 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "scale-in": "scale-in 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-up": "slide-in-up 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-right": "slide-in-right 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-left": "slide-in-left 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        shimmer: "shimmer 1.5s ease-in-out infinite",
        spin: "spin 1s linear infinite",
        "spin-slow": "spin 3s linear infinite",
        pulse: "pulse 2s ease-in-out infinite",
        shake: "shake 0.5s ease-in-out",
        "modal-enter": "modal-enter 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "modal-exit": "modal-exit 200ms ease-in forwards",
        "toast-enter": "toast-enter 300ms cubic-bezier(0.34, 1.56, 0.64, 1) forwards",
        "toast-exit": "toast-exit 200ms ease-in forwards",
        bounce: "bounce 1s infinite",
        wiggle: "wiggle 0.3s ease-in-out",
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
      },
      
      // ========================================================================
      // BACKGROUNDS
      // ========================================================================
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "gradient-navy": "linear-gradient(135deg, #131D4F 0%, #254D70 100%)",
        "gradient-terra": "linear-gradient(135deg, #954C2E 0%, #B86A4A 100%)",
        "gradient-cream": "linear-gradient(180deg, #FAF7F2 0%, #EFE4D2 100%)",
        shimmer: "linear-gradient(90deg, var(--color-cream-100) 0%, var(--color-cream-50) 50%, var(--color-cream-100) 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
