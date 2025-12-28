'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SectionSpacing = 'none' | 'sm' | 'md' | 'lg' | 'xl';
export type SectionBackground = 'default' | 'alt' | 'white' | 'transparent';

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

export default Section;
