'use client';

import React, { HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Width (CSS value or Tailwind class) */
  width?: string;
  /** Height (CSS value or Tailwind class) */
  height?: string;
  /** Border radius */
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  /** Animation enabled */
  animate?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const roundedStyles = {
  none: 'rounded-none',
  sm: 'rounded-sm',
  md: 'rounded-md',
  lg: 'rounded-lg',
  full: 'rounded-full',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Skeleton: React.FC<SkeletonProps> = ({
  width,
  height,
  rounded = 'md',
  animate = true,
  className = '',
  style,
  ...props
}) => {
  return (
    <div
      className={`
        bg-gradient-to-r from-cream-100 via-cream-50 to-cream-100
        bg-[length:200%_100%]
        ${animate ? 'animate-shimmer' : ''}
        ${roundedStyles[rounded]}
        ${width?.startsWith('w-') ? width : ''}
        ${height?.startsWith('h-') ? height : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      style={{
        width: width && !width.startsWith('w-') ? width : undefined,
        height: height && !height.startsWith('h-') ? height : undefined,
        ...style,
      }}
      aria-hidden="true"
      {...props}
    />
  );
};

// =============================================================================
// SKELETON TEXT
// =============================================================================

export interface SkeletonTextProps extends HTMLAttributes<HTMLDivElement> {
  /** Number of lines */
  lines?: number;
  /** Width of last line (percentage) */
  lastLineWidth?: number;
  /** Line spacing */
  spacing?: 'sm' | 'md' | 'lg';
}

export const SkeletonText: React.FC<SkeletonTextProps> = ({
  lines = 3,
  lastLineWidth = 60,
  spacing = 'md',
  className = '',
  ...props
}) => {
  const spacingStyles = {
    sm: 'space-y-2',
    md: 'space-y-3',
    lg: 'space-y-4',
  };

  return (
    <div className={`${spacingStyles[spacing]} ${className}`} {...props}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          width={i === lines - 1 ? `${lastLineWidth}%` : '100%'}
          height="1em"
          rounded="sm"
        />
      ))}
    </div>
  );
};

// =============================================================================
// SKELETON AVATAR
// =============================================================================

export interface SkeletonAvatarProps extends HTMLAttributes<HTMLDivElement> {
  /** Avatar size */
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export const SkeletonAvatar: React.FC<SkeletonAvatarProps> = ({
  size = 'md',
  className = '',
  ...props
}) => {
  const sizeStyles = {
    sm: 'w-8 h-8',
    md: 'w-10 h-10',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  };

  return (
    <Skeleton
      className={`${sizeStyles[size]} ${className}`}
      rounded="full"
      {...props}
    />
  );
};

// =============================================================================
// SKELETON CARD
// =============================================================================

export interface SkeletonCardProps extends HTMLAttributes<HTMLDivElement> {
  /** Show avatar */
  hasAvatar?: boolean;
  /** Number of text lines */
  lines?: number;
  /** Show image placeholder */
  hasImage?: boolean;
}

export const SkeletonCard: React.FC<SkeletonCardProps> = ({
  hasAvatar = false,
  lines = 3,
  hasImage = false,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`bg-white rounded-lg shadow-md p-6 ${className}`}
      {...props}
    >
      {hasImage && (
        <Skeleton
          width="100%"
          height="160px"
          rounded="md"
          className="mb-4 -mx-6 -mt-6 w-[calc(100%+3rem)]"
        />
      )}
      
      {hasAvatar && (
        <div className="flex items-center gap-3 mb-4">
          <SkeletonAvatar size="md" />
          <div className="flex-1 space-y-2">
            <Skeleton width="40%" height="1em" rounded="sm" />
            <Skeleton width="60%" height="0.875em" rounded="sm" />
          </div>
        </div>
      )}
      
      {!hasAvatar && (
        <Skeleton width="60%" height="1.5em" rounded="sm" className="mb-4" />
      )}
      
      <SkeletonText lines={lines} spacing="sm" />
    </div>
  );
};

// =============================================================================
// SKELETON TABLE
// =============================================================================

export interface SkeletonTableProps extends HTMLAttributes<HTMLDivElement> {
  /** Number of rows */
  rows?: number;
  /** Number of columns */
  columns?: number;
}

export const SkeletonTable: React.FC<SkeletonTableProps> = ({
  rows = 5,
  columns = 4,
  className = '',
  ...props
}) => {
  return (
    <div className={`overflow-hidden rounded-lg border border-border-light ${className}`} {...props}>
      {/* Header */}
      <div className="flex gap-4 p-4 bg-cream-50 border-b border-border-light">
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton
            key={`header-${i}`}
            width={i === 0 ? '30%' : '20%'}
            height="1em"
            rounded="sm"
          />
        ))}
      </div>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div
          key={`row-${rowIndex}`}
          className="flex gap-4 p-4 border-b border-border-light last:border-b-0"
        >
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton
              key={`cell-${rowIndex}-${colIndex}`}
              width={colIndex === 0 ? '30%' : '20%'}
              height="1em"
              rounded="sm"
            />
          ))}
        </div>
      ))}
    </div>
  );
};

export default Skeleton;
