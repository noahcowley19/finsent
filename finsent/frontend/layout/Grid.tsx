'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type GridCols = 1 | 2 | 3 | 4 | 5 | 6 | 12;
export type GridGap = 'none' | 'sm' | 'md' | 'lg' | 'xl';

export interface GridProps extends HTMLAttributes<HTMLDivElement> {
  /** Number of columns */
  cols?: GridCols;
  /** Number of columns on small screens */
  colsSm?: GridCols;
  /** Number of columns on medium screens */
  colsMd?: GridCols;
  /** Number of columns on large screens */
  colsLg?: GridCols;
  /** Gap between items */
  gap?: GridGap;
  /** Content */
  children: ReactNode;
}

export interface GridItemProps extends HTMLAttributes<HTMLDivElement> {
  /** Column span */
  span?: 1 | 2 | 3 | 4 | 5 | 6 | 12 | 'full';
  /** Column span on small screens */
  spanSm?: 1 | 2 | 3 | 4 | 5 | 6 | 12 | 'full';
  /** Column span on medium screens */
  spanMd?: 1 | 2 | 3 | 4 | 5 | 6 | 12 | 'full';
  /** Column span on large screens */
  spanLg?: 1 | 2 | 3 | 4 | 5 | 6 | 12 | 'full';
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const colStyles: Record<GridCols, string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
  5: 'grid-cols-5',
  6: 'grid-cols-6',
  12: 'grid-cols-12',
};

const colSmStyles: Record<GridCols, string> = {
  1: 'sm:grid-cols-1',
  2: 'sm:grid-cols-2',
  3: 'sm:grid-cols-3',
  4: 'sm:grid-cols-4',
  5: 'sm:grid-cols-5',
  6: 'sm:grid-cols-6',
  12: 'sm:grid-cols-12',
};

const colMdStyles: Record<GridCols, string> = {
  1: 'md:grid-cols-1',
  2: 'md:grid-cols-2',
  3: 'md:grid-cols-3',
  4: 'md:grid-cols-4',
  5: 'md:grid-cols-5',
  6: 'md:grid-cols-6',
  12: 'md:grid-cols-12',
};

const colLgStyles: Record<GridCols, string> = {
  1: 'lg:grid-cols-1',
  2: 'lg:grid-cols-2',
  3: 'lg:grid-cols-3',
  4: 'lg:grid-cols-4',
  5: 'lg:grid-cols-5',
  6: 'lg:grid-cols-6',
  12: 'lg:grid-cols-12',
};

const gapStyles: Record<GridGap, string> = {
  none: 'gap-0',
  sm: 'gap-4',
  md: 'gap-6',
  lg: 'gap-8',
  xl: 'gap-12',
};

const spanStyles: Record<string, string> = {
  '1': 'col-span-1',
  '2': 'col-span-2',
  '3': 'col-span-3',
  '4': 'col-span-4',
  '5': 'col-span-5',
  '6': 'col-span-6',
  '12': 'col-span-12',
  'full': 'col-span-full',
};

const spanSmStyles: Record<string, string> = {
  '1': 'sm:col-span-1',
  '2': 'sm:col-span-2',
  '3': 'sm:col-span-3',
  '4': 'sm:col-span-4',
  '5': 'sm:col-span-5',
  '6': 'sm:col-span-6',
  '12': 'sm:col-span-12',
  'full': 'sm:col-span-full',
};

const spanMdStyles: Record<string, string> = {
  '1': 'md:col-span-1',
  '2': 'md:col-span-2',
  '3': 'md:col-span-3',
  '4': 'md:col-span-4',
  '5': 'md:col-span-5',
  '6': 'md:col-span-6',
  '12': 'md:col-span-12',
  'full': 'md:col-span-full',
};

const spanLgStyles: Record<string, string> = {
  '1': 'lg:col-span-1',
  '2': 'lg:col-span-2',
  '3': 'lg:col-span-3',
  '4': 'lg:col-span-4',
  '5': 'lg:col-span-5',
  '6': 'lg:col-span-6',
  '12': 'lg:col-span-12',
  'full': 'lg:col-span-full',
};

// =============================================================================
// GRID COMPONENT
// =============================================================================

export const Grid: React.FC<GridProps> = ({
  cols = 1,
  colsSm,
  colsMd,
  colsLg,
  gap = 'md',
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        grid
        ${colStyles[cols]}
        ${colsSm ? colSmStyles[colsSm] : ''}
        ${colsMd ? colMdStyles[colsMd] : ''}
        ${colsLg ? colLgStyles[colsLg] : ''}
        ${gapStyles[gap]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// GRID ITEM COMPONENT
// =============================================================================

export const GridItem: React.FC<GridItemProps> = ({
  span,
  spanSm,
  spanMd,
  spanLg,
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        ${span ? spanStyles[String(span)] : ''}
        ${spanSm ? spanSmStyles[String(spanSm)] : ''}
        ${spanMd ? spanMdStyles[String(spanMd)] : ''}
        ${spanLg ? spanLgStyles[String(spanLg)] : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// FLEX LAYOUT HELPERS
// =============================================================================

export interface StackProps extends HTMLAttributes<HTMLDivElement> {
  /** Direction */
  direction?: 'vertical' | 'horizontal';
  /** Gap between items */
  gap?: GridGap;
  /** Alignment */
  align?: 'start' | 'center' | 'end' | 'stretch';
  /** Justify content */
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  /** Wrap items */
  wrap?: boolean;
  /** Content */
  children: ReactNode;
}

export const Stack: React.FC<StackProps> = ({
  direction = 'vertical',
  gap = 'md',
  align = 'stretch',
  justify = 'start',
  wrap = false,
  children,
  className = '',
  ...props
}) => {
  const alignStyles = {
    start: 'items-start',
    center: 'items-center',
    end: 'items-end',
    stretch: 'items-stretch',
  };

  const justifyStyles = {
    start: 'justify-start',
    center: 'justify-center',
    end: 'justify-end',
    between: 'justify-between',
    around: 'justify-around',
  };

  return (
    <div
      className={`
        flex
        ${direction === 'horizontal' ? 'flex-row' : 'flex-col'}
        ${gapStyles[gap]}
        ${alignStyles[align]}
        ${justifyStyles[justify]}
        ${wrap ? 'flex-wrap' : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// TWO COLUMN LAYOUT
// =============================================================================

export interface TwoColumnProps extends HTMLAttributes<HTMLDivElement> {
  /** Left column (or main content) */
  main: ReactNode;
  /** Right column (or sidebar) */
  sidebar: ReactNode;
  /** Sidebar position */
  sidebarPosition?: 'left' | 'right';
  /** Sidebar width */
  sidebarWidth?: 'narrow' | 'medium' | 'wide';
  /** Gap between columns */
  gap?: GridGap;
  /** Stack on mobile */
  stackOnMobile?: boolean;
}

export const TwoColumn: React.FC<TwoColumnProps> = ({
  main,
  sidebar,
  sidebarPosition = 'right',
  sidebarWidth = 'medium',
  gap = 'lg',
  stackOnMobile = true,
  className = '',
  ...props
}) => {
  const sidebarWidths = {
    narrow: 'lg:w-64',
    medium: 'lg:w-80',
    wide: 'lg:w-96',
  };

  return (
    <div
      className={`
        flex
        ${stackOnMobile ? 'flex-col lg:flex-row' : 'flex-row'}
        ${gapStyles[gap]}
        ${className}
      `}
      {...props}
    >
      {sidebarPosition === 'left' && (
        <aside className={`flex-shrink-0 ${sidebarWidths[sidebarWidth]}`}>
          {sidebar}
        </aside>
      )}
      <main className="flex-1 min-w-0">{main}</main>
      {sidebarPosition === 'right' && (
        <aside className={`flex-shrink-0 ${sidebarWidths[sidebarWidth]}`}>
          {sidebar}
        </aside>
      )}
    </div>
  );
};

export default Grid;
