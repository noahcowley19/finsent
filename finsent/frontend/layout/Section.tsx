'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SectionBackground = 'default' | 'alt' | 'dark' | 'gradient';
export type SectionSpacing = 'none' | 'sm' | 'md' | 'lg' | 'xl';

export interface SectionProps extends HTMLAttributes<HTMLElement> {
  /** Background style */
  background?: SectionBackground;
  /** Vertical padding */
  spacing?: SectionSpacing;
  /** Include container wrapper */
  container?: boolean;
  /** Container size (if container is true) */
  containerSize?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full';
  /** HTML element to render as */
  as?: 'section' | 'div' | 'article';
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const backgroundStyles: Record<SectionBackground, string> = {
  default: 'bg-cream-50',
  alt: 'bg-cream-100',
  dark: 'bg-navy-900 text-white',
  gradient: 'bg-gradient-to-br from-cream-50 to-cream-100',
};

const spacingStyles: Record<SectionSpacing, string> = {
  none: 'py-0',
  sm: 'py-8 lg:py-12',
  md: 'py-12 lg:py-16',
  lg: 'py-16 lg:py-24',
  xl: 'py-24 lg:py-32',
};

const containerSizes = {
  xs: 'max-w-xl',
  sm: 'max-w-2xl',
  md: 'max-w-4xl',
  lg: 'max-w-6xl',
  xl: 'max-w-7xl',
  full: 'max-w-full',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Section: React.FC<SectionProps> = ({
  background = 'default',
  spacing = 'lg',
  container = true,
  containerSize = 'xl',
  as: Component = 'section',
  children,
  className = '',
  ...props
}) => {
  return (
    <Component
      className={`
        ${backgroundStyles[background]}
        ${spacingStyles[spacing]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {container ? (
        <div className={`${containerSizes[containerSize]} mx-auto px-4 sm:px-6 lg:px-8`}>
          {children}
        </div>
      ) : (
        children
      )}
    </Component>
  );
};

// =============================================================================
// SECTION HEADER
// =============================================================================

export interface SectionHeaderProps extends HTMLAttributes<HTMLDivElement> {
  /** Section title */
  title: string;
  /** Section subtitle/description */
  subtitle?: string;
  /** Overline text (small text above title) */
  overline?: string;
  /** Alignment */
  align?: 'left' | 'center';
  /** Actions (buttons, links) */
  actions?: ReactNode;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  overline,
  align = 'left',
  actions,
  className = '',
  ...props
}) => {
  const alignStyles = {
    left: 'text-left',
    center: 'text-center mx-auto',
  };

  return (
    <div
      className={`
        mb-8 lg:mb-12 max-w-3xl
        ${alignStyles[align]}
        ${className}
      `}
      {...props}
    >
      {overline && (
        <p className="text-overline text-terra-500 uppercase tracking-widest mb-3">
          {overline}
        </p>
      )}
      <div className={`flex ${align === 'center' ? 'flex-col items-center' : 'flex-col sm:flex-row sm:items-end sm:justify-between'} gap-4`}>
        <div>
          <h2 className="font-heading text-heading-xl lg:text-display-sm text-navy-900">
            {title}
          </h2>
          {subtitle && (
            <p className="mt-2 text-body-md lg:text-body-lg text-neutral-600">
              {subtitle}
            </p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-3 flex-shrink-0">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// HERO SECTION
// =============================================================================

export interface HeroSectionProps extends HTMLAttributes<HTMLElement> {
  /** Full viewport height */
  fullHeight?: boolean;
  /** Background variant */
  background?: 'light' | 'dark' | 'gradient';
  /** Content alignment */
  align?: 'left' | 'center';
  /** Content */
  children: ReactNode;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  fullHeight = false,
  background = 'light',
  align = 'left',
  children,
  className = '',
  ...props
}) => {
  const bgStyles = {
    light: 'bg-cream-50',
    dark: 'bg-navy-900 text-white',
    gradient: 'bg-gradient-to-br from-cream-50 via-cream-100 to-terra-50',
  };

  return (
    <section
      className={`
        relative
        ${bgStyles[background]}
        ${fullHeight ? 'min-h-[calc(100vh-5rem)]' : 'py-16 lg:py-24'}
        ${className}
      `}
      {...props}
    >
      <div className={`
        max-w-7xl mx-auto px-4 sm:px-6 lg:px-8
        ${fullHeight ? 'h-full flex items-center' : ''}
        ${align === 'center' ? 'text-center' : ''}
      `}>
        {children}
      </div>
    </section>
  );
};

export default Section;
