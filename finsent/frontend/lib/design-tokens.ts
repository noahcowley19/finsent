/* ============================================================================
   CAVERAY DESIGN TOKENS (TypeScript)
   ============================================================================
   This file provides type-safe access to design tokens in JavaScript/TypeScript.
   Use these constants when you need design values in JS (e.g., Canvas rendering,
   dynamic styles, or component props).
   
   Location: /lib/design-tokens.ts
   ============================================================================ */

// =============================================================================
// COLOR PALETTE
// =============================================================================

export const colors = {
  // Primary: Cream (Backgrounds)
  cream: {
    50: '#FAF7F2',   // Page background
    100: '#EFE4D2',  // Card backgrounds
    200: '#E5D9C3',  // Hover states, dividers
    300: '#D4C4A8',  // Disabled states
    400: '#C4B08D',  // Subtle accents
  },
  
  // Primary: Navy (Brand, Text)
  navy: {
    50: '#E8F1F8',
    100: '#B8D1E5',
    200: '#7BA3C2',
    300: '#4A7A9D',
    400: '#3A6285',
    500: '#254D70',  // Interactive elements
    600: '#2A3F6A',
    700: '#1E2E5A',  // Body text
    800: '#1A2759',
    900: '#131D4F',  // Headings
  },
  
  // Accent: Terracotta (CTAs)
  terra: {
    50: '#FBF3EF',
    100: '#F5E6DE',
    200: '#E8BBA8',
    300: '#D4917A',
    400: '#B86A4A',
    500: '#954C2E',  // Primary buttons
    600: '#7A3D24',  // Hover state
    700: '#6B3318',  // Active state
  },
  
  // Semantic: Success
  success: {
    50: '#F0FAF4',
    100: '#E8F5ED',
    200: '#A3D9B8',
    300: '#6FBD8F',
    400: '#4A9D6F',
    500: '#2D7A4F',  // Primary
    600: '#246B44',
    700: '#1D5A38',
  },
  
  // Semantic: Warning
  warning: {
    50: '#FFFBF0',
    100: '#FDF6E3',
    200: '#E8CDA0',
    300: '#D4A85C',
    400: '#B8893D',
    500: '#9A6B28',  // Primary
    600: '#8A5F1F',
    700: '#7A5319',
  },
  
  // Semantic: Error
  error: {
    50: '#FEF5F5',
    100: '#FDEBEB',
    200: '#F5B3B3',
    300: '#E88080',
    400: '#D45A5A',
    500: '#B83A3A',  // Primary
    600: '#A12F2F',
    700: '#8A2525',
  },
  
  // Neutral (Grays)
  neutral: {
    50: '#F9FAFB',
    100: '#F3F4F6',
    200: '#E5E7EB',
    300: '#C4C4C4',
    400: '#9CA3AF',
    500: '#6B6B6B',
    600: '#525252',
    700: '#404040',
    800: '#2D2D2D',
    900: '#1A1A1A',
  },
  
  // Special
  white: '#FFFFFF',
  black: '#000000',
  transparent: 'transparent',
} as const;

// Semantic color aliases
export const semanticColors = {
  bg: {
    primary: colors.cream[50],
    secondary: colors.cream[100],
    tertiary: colors.cream[200],
    elevated: colors.white,
    inverse: colors.navy[900],
  },
  text: {
    primary: colors.navy[900],
    secondary: colors.navy[700],
    tertiary: colors.neutral[600],
    muted: colors.neutral[400],
    inverse: colors.white,
  },
  border: {
    light: 'rgba(19, 29, 79, 0.08)',
    medium: 'rgba(19, 29, 79, 0.12)',
    heavy: 'rgba(19, 29, 79, 0.20)',
    focus: colors.navy[500],
  },
  link: {
    default: colors.navy[500],
    hover: colors.navy[700],
    visited: colors.navy[600],
  },
} as const;

// =============================================================================
// TYPOGRAPHY
// =============================================================================

export const fonts = {
  display: "'Fraunces', Georgia, 'Times New Roman', serif",
  heading: "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  body: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
} as const;

export const fontSizes = {
  displayXl: '4.5rem',    // 72px
  displayLg: '3.5rem',    // 56px
  displayMd: '2.5rem',    // 40px
  displaySm: '2rem',      // 32px
  headingXl: '1.75rem',   // 28px
  headingLg: '1.5rem',    // 24px
  headingMd: '1.25rem',   // 20px
  headingSm: '1.125rem',  // 18px
  bodyLg: '1.125rem',     // 18px
  bodyMd: '1rem',         // 16px
  bodySm: '0.875rem',     // 14px
  caption: '0.75rem',     // 12px
  overline: '0.75rem',    // 12px
} as const;

// Pixel values for Canvas rendering
export const fontSizesPx = {
  displayXl: 72,
  displayLg: 56,
  displayMd: 40,
  displaySm: 32,
  headingXl: 28,
  headingLg: 24,
  headingMd: 20,
  headingSm: 18,
  bodyLg: 18,
  bodyMd: 16,
  bodySm: 14,
  caption: 12,
  overline: 12,
} as const;

export const lineHeights = {
  none: 1,
  tight: 1.15,
  snug: 1.25,
  normal: 1.5,
  relaxed: 1.625,
  loose: 1.75,
} as const;

export const letterSpacing = {
  tighter: '-0.03em',
  tight: '-0.02em',
  normal: '0',
  wide: '0.01em',
  wider: '0.05em',
  widest: '0.1em',
} as const;

export const fontWeights = {
  light: 300,
  normal: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
  extrabold: 800,
} as const;

// =============================================================================
// SPACING
// =============================================================================

export const spacing = {
  0: '0',
  px: '1px',
  0.5: '0.125rem',   // 2px
  1: '0.25rem',      // 4px
  1.5: '0.375rem',   // 6px
  2: '0.5rem',       // 8px
  2.5: '0.625rem',   // 10px
  3: '0.75rem',      // 12px
  3.5: '0.875rem',   // 14px
  4: '1rem',         // 16px
  5: '1.25rem',      // 20px
  6: '1.5rem',       // 24px
  7: '1.75rem',      // 28px
  8: '2rem',         // 32px
  9: '2.25rem',      // 36px
  10: '2.5rem',      // 40px
  11: '2.75rem',     // 44px
  12: '3rem',        // 48px
  14: '3.5rem',      // 56px
  16: '4rem',        // 64px
  20: '5rem',        // 80px
  24: '6rem',        // 96px
  28: '7rem',        // 112px
  32: '8rem',        // 128px
  36: '9rem',        // 144px
  40: '10rem',       // 160px
  44: '11rem',       // 176px
  48: '12rem',       // 192px
} as const;

// Pixel values for Canvas rendering
export const spacingPx = {
  0: 0,
  px: 1,
  0.5: 2,
  1: 4,
  1.5: 6,
  2: 8,
  2.5: 10,
  3: 12,
  3.5: 14,
  4: 16,
  5: 20,
  6: 24,
  7: 28,
  8: 32,
  9: 36,
  10: 40,
  11: 44,
  12: 48,
  14: 56,
  16: 64,
  20: 80,
  24: 96,
  28: 112,
  32: 128,
} as const;

// =============================================================================
// SIZING
// =============================================================================

export const containers = {
  xs: '20rem',    // 320px
  sm: '24rem',    // 384px
  md: '28rem',    // 448px
  lg: '32rem',    // 512px
  xl: '36rem',    // 576px
  '2xl': '42rem', // 672px
  '3xl': '48rem', // 768px
  '4xl': '56rem', // 896px
  '5xl': '64rem', // 1024px
  '6xl': '72rem', // 1152px
  '7xl': '80rem', // 1280px
  max: '90rem',   // 1440px
} as const;

export const componentHeights = {
  inputSm: '2rem',      // 32px
  inputMd: '2.5rem',    // 40px
  inputLg: '3rem',      // 48px
  inputXl: '3.5rem',    // 56px
  buttonSm: '2rem',     // 32px
  buttonMd: '2.5rem',   // 40px
  buttonLg: '3rem',     // 48px
  buttonXl: '3.5rem',   // 56px
  nav: '5rem',          // 80px
  navScrolled: '4rem',  // 64px
} as const;

// =============================================================================
// BORDERS & RADIUS
// =============================================================================

export const borderRadius = {
  none: '0',
  sm: '0.375rem',   // 6px
  md: '0.5rem',     // 8px
  lg: '0.75rem',    // 12px
  xl: '1rem',       // 16px
  '2xl': '1.5rem',  // 24px
  '3xl': '2rem',    // 32px
  full: '9999px',
} as const;

export const borderRadiusPx = {
  none: 0,
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
  '2xl': 24,
  '3xl': 32,
  full: 9999,
} as const;

// =============================================================================
// SHADOWS
// =============================================================================

export const shadows = {
  xs: '0 1px 2px rgba(19, 29, 79, 0.04)',
  sm: '0 2px 4px rgba(19, 29, 79, 0.06)',
  md: '0 4px 12px rgba(19, 29, 79, 0.08)',
  lg: '0 8px 24px rgba(19, 29, 79, 0.10)',
  xl: '0 16px 48px rgba(19, 29, 79, 0.12)',
  '2xl': '0 24px 64px rgba(19, 29, 79, 0.16)',
  inner: 'inset 0 2px 4px rgba(19, 29, 79, 0.04)',
  none: '0 0 0 0 transparent',
  terra: '0 4px 14px rgba(149, 76, 46, 0.25)',
  terraLg: '0 8px 24px rgba(149, 76, 46, 0.30)',
  success: '0 4px 14px rgba(45, 122, 79, 0.25)',
  error: '0 4px 14px rgba(184, 58, 58, 0.25)',
} as const;

// =============================================================================
// ANIMATION
// =============================================================================

export const durations = {
  instant: '100ms',
  fast: '200ms',
  normal: '300ms',
  slow: '400ms',
  slower: '600ms',
  slowest: '800ms',
} as const;

export const durationsMs = {
  instant: 100,
  fast: 200,
  normal: 300,
  slow: 400,
  slower: 600,
  slowest: 800,
} as const;

export const easings = {
  linear: 'linear',
  in: 'cubic-bezier(0.4, 0, 1, 1)',
  out: 'cubic-bezier(0, 0, 0.2, 1)',
  inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
  outExpo: 'cubic-bezier(0.16, 1, 0.3, 1)',
  outBack: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
  bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
} as const;

// =============================================================================
// Z-INDEX
// =============================================================================

export const zIndex = {
  below: -1,
  base: 0,
  above: 1,
  dropdown: 10,
  sticky: 20,
  fixed: 30,
  modalBackdrop: 40,
  modal: 50,
  popover: 60,
  tooltip: 70,
  toast: 80,
  max: 9999,
} as const;

// =============================================================================
// BREAKPOINTS
// =============================================================================

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;

export const breakpointsPx = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// =============================================================================
// CHART-SPECIFIC TOKENS (for Canvas rendering)
// =============================================================================

export const chartColors = {
  primary: colors.navy[500],
  secondary: colors.terra[400],
  positive: colors.success[500],
  negative: colors.error[500],
  neutral: colors.neutral[400],
  grid: colors.neutral[200],
  axis: colors.neutral[600],
  tooltip: {
    bg: colors.cream[50],
    border: semanticColors.border.light,
    text: colors.navy[900],
  },
  candlestick: {
    bullish: colors.success[500],
    bearish: colors.error[500],
    wick: colors.neutral[600],
  },
} as const;

export const chartDimensions = {
  padding: {
    top: 20,
    right: 20,
    bottom: 40,
    left: 60,
  },
  fontSize: {
    axis: 12,
    label: 14,
    value: 16,
  },
  lineWidth: {
    thin: 1,
    normal: 2,
    thick: 3,
  },
  pointRadius: {
    small: 3,
    normal: 4,
    large: 6,
  },
} as const;

// =============================================================================
// TYPE EXPORTS
// =============================================================================

export type ColorKey = keyof typeof colors;
export type SpacingKey = keyof typeof spacing;
export type FontSizeKey = keyof typeof fontSizes;
export type BorderRadiusKey = keyof typeof borderRadius;
export type ShadowKey = keyof typeof shadows;
export type DurationKey = keyof typeof durations;
export type EasingKey = keyof typeof easings;
export type BreakpointKey = keyof typeof breakpoints;
export type ZIndexKey = keyof typeof zIndex;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Get a color value by path (e.g., 'navy.500')
 */
export function getColor(path: string): string {
  const [palette, shade] = path.split('.');
  const paletteObj = colors[palette as keyof typeof colors];
  if (typeof paletteObj === 'string') return paletteObj;
  return paletteObj?.[shade as keyof typeof paletteObj] || path;
}

/**
 * Convert rem to pixels (assumes 16px base)
 */
export function remToPx(rem: string | number): number {
  const value = typeof rem === 'string' ? parseFloat(rem) : rem;
  return value * 16;
}

/**
 * Convert pixels to rem (assumes 16px base)
 */
export function pxToRem(px: number): string {
  return `${px / 16}rem`;
}

/**
 * Create a CSS transition string
 */
export function createTransition(
  properties: string[],
  duration: keyof typeof durations = 'fast',
  easing: keyof typeof easings = 'out'
): string {
  return properties
    .map(prop => `${prop} ${durations[duration]} ${easings[easing]}`)
    .join(', ');
}

/**
 * Get responsive value based on breakpoint
 */
export function getResponsiveValue<T>(
  values: Partial<Record<keyof typeof breakpoints | 'base', T>>,
  currentWidth: number
): T | undefined {
  const sortedBreakpoints = Object.entries(breakpointsPx)
    .sort(([, a], [, b]) => b - a);
  
  for (const [key, minWidth] of sortedBreakpoints) {
    if (currentWidth >= minWidth && values[key as keyof typeof breakpoints]) {
      return values[key as keyof typeof breakpoints];
    }
  }
  
  return values.base;
}
