'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SectionSpacing = 'none' | 'sm' | 'md' | 'lg' | 'xl';
export type SectionBackground = 'default' | 'alt' | 'white' | 'transparent' | 'gradient';

export interface SectionProps extends HTMLAttributes<HTMLElement> {
  /** Content */
  children: ReactNode;
  /** Vertical spacing */
  spacing?: SectionSpacing;
  /** Background style */
  background?: SectionBackground;
  /** Use container max-width */
  container?: boolean;
  /** HTML element */
  as?: 'section' | 'div' | 'article';
}

// =============================================================================
// STYLES
// =============================================================================

const spacingStyles: Record<SectionSpacing, string> = {
  none: 'py-0',
  sm: 'py-8 lg:py-12',
  md: 'py-12 lg:py-16',
  lg: 'py-16 lg:py-24',
  xl: 'py-20 lg:py-32',
};

const backgroundStyles: Record<SectionBackground, string> = {
  default: 'bg-ink-50',
  alt: 'bg-ink-100',
  white: 'bg-white',
  transparent: 'bg-transparent',
  gradient: 'bg-gradient-to-br from-cream-50 via-white to-cream-100',
};

// =============================================================================
// SECTION COMPONENT
// =============================================================================

export const Section: React.FC<SectionProps> = ({
  children,
  spacing = 'md',
  background = 'default',
  container = true,
  as: Component = 'section',
  className = '',
  ...props
}) => {
  return (
    <Component
      className={`
        ${spacingStyles[spacing]}
        ${backgroundStyles[background]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {container ? (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      ) : (
        children
      )}
    </Component>
  );
};

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  align = 'left',
  className = '',
}) => {
  const alignmentClasses: Record<string, string> = {
    left: 'text-left',
    center: 'text-center mx-auto',
    right: 'text-right ml-auto',
  };

  return (
    <div className={`mb-8 lg:mb-12 max-w-3xl ${alignmentClasses[align]} ${className}`}>
      {title && (
        <h2 className="text-display-sm lg:text-display-md font-display text-navy-900 mb-4">
          {title}
        </h2>
      )}
      {subtitle && (
        <p className="text-body-lg text-neutral-600">
          {subtitle}
        </p>
      )}
    </div>
  );
};

export interface SectionHeaderProps {
  title?: string;
  subtitle?: string;
  align?: 'left' | 'center' | 'right';
  className?: string;
}

// =============================================================================
// HERO SECTION
// =============================================================================

export interface HeroSectionProps extends SectionProps {
  gradient?: boolean;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  children,
  gradient = true,
  className = '',
  ...props
}) => {
  return (
    <Section
      spacing="xl"
      background={gradient ? 'gradient' : 'white'}
      className={`relative overflow-hidden ${className}`}
      {...props}
    >
      <div className="relative z-10">
        {children}
      </div>

      {/* Subtle atmospheric blobs */}
      <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 w-[500px] h-[500px] bg-terra-100/30 rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 translate-y-1/2 -translate-x-1/4 w-[400px] h-[400px] bg-warning-100/20 rounded-full blur-3xl" />
    </Section>
  );
};

export default Section;
