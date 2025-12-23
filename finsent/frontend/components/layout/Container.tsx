'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type ContainerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface ContainerProps extends HTMLAttributes<HTMLDivElement> {
  /** Maximum width */
  size?: ContainerSize;
  /** Center container */
  centered?: boolean;
  /** Add horizontal padding */
  padding?: boolean;
  /** HTML element to render as */
  as?: 'div' | 'section' | 'article' | 'main';
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<ContainerSize, string> = {
  xs: 'max-w-xl',         // 576px
  sm: 'max-w-2xl',        // 672px
  md: 'max-w-4xl',        // 896px
  lg: 'max-w-6xl',        // 1152px
  xl: 'max-w-7xl',        // 1280px
  full: 'max-w-full',     // 100%
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Container: React.FC<ContainerProps> = ({
  size = 'xl',
  centered = true,
  padding = true,
  as: Component = 'div',
  children,
  className = '',
  ...props
}) => {
  return (
    <Component
      className={`
        w-full
        ${sizeStyles[size]}
        ${centered ? 'mx-auto' : ''}
        ${padding ? 'px-4 sm:px-6 lg:px-8' : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </Component>
  );
};

// =============================================================================
// NARROW CONTAINER (for forms, auth pages, etc.)
// =============================================================================

export interface NarrowContainerProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const NarrowContainer: React.FC<NarrowContainerProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <Container size="xs" className={className} {...props}>
      {children}
    </Container>
  );
};

// =============================================================================
// CONTENT CONTAINER (for readable content)
// =============================================================================

export interface ContentContainerProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const ContentContainer: React.FC<ContentContainerProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <Container size="md" className={`prose prose-navy ${className}`} {...props}>
      {children}
    </Container>
  );
};

export default Container;
