'use client';

import React, { useEffect, useCallback, ReactNode, HTMLAttributes } from 'react';
import { createPortal } from 'react-dom';

// =============================================================================
// TYPES
// =============================================================================

export type DrawerPosition = 'left' | 'right' | 'top' | 'bottom';
export type DrawerSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface DrawerProps extends HTMLAttributes<HTMLDivElement> {
  /** Whether drawer is open */
  isOpen: boolean;
  /** Called when drawer should close */
  onClose: () => void;
  /** Drawer position */
  position?: DrawerPosition;
  /** Drawer size */
  size?: DrawerSize;
  /** Drawer title */
  title?: string;
  /** Close on backdrop click */
  closeOnBackdrop?: boolean;
  /** Close on escape key */
  closeOnEscape?: boolean;
  /** Show close button */
  showCloseButton?: boolean;
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const positionStyles: Record<DrawerPosition, string> = {
  left: 'left-0 top-0 h-full',
  right: 'right-0 top-0 h-full',
  top: 'top-0 left-0 w-full',
  bottom: 'bottom-0 left-0 w-full',
};

const sizeStylesHorizontal: Record<DrawerSize, string> = {
  sm: 'w-72',
  md: 'w-96',
  lg: 'w-[32rem]',
  xl: 'w-[42rem]',
  full: 'w-full',
};

const sizeStylesVertical: Record<DrawerSize, string> = {
  sm: 'h-48',
  md: 'h-72',
  lg: 'h-96',
  xl: 'h-[32rem]',
  full: 'h-full',
};

const animationStyles: Record<DrawerPosition, { enter: string; base: string }> = {
  left: {
    enter: 'animate-slide-in-left',
    base: '-translate-x-full',
  },
  right: {
    enter: 'animate-slide-in-right',
    base: 'translate-x-full',
  },
  top: {
    enter: 'animate-[slideInDown_400ms_cubic-bezier(0.16,1,0.3,1)_forwards]',
    base: '-translate-y-full',
  },
  bottom: {
    enter: 'animate-slide-in-up',
    base: 'translate-y-full',
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Drawer: React.FC<DrawerProps> = ({
  isOpen,
  onClose,
  position = 'right',
  size = 'md',
  title,
  closeOnBackdrop = true,
  closeOnEscape = true,
  showCloseButton = true,
  children,
  className = '',
  ...props
}) => {
  const isHorizontal = position === 'left' || position === 'right';

  // Handle escape key
  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && closeOnEscape) {
        onClose();
      }
    },
    [closeOnEscape, onClose]
  );

  // Handle backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget && closeOnBackdrop) {
        onClose();
      }
    },
    [closeOnBackdrop, onClose]
  );

  // Setup/cleanup
  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleEscape]);

  // Don't render if closed
  if (!isOpen) return null;

  const drawerContent = (
    <div
      className="fixed inset-0 z-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'drawer-title' : undefined}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-navy-900/40 backdrop-blur-sm animate-fade-in"
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Drawer content */}
      <div
        className={`
          fixed
          ${positionStyles[position]}
          ${isHorizontal ? sizeStylesHorizontal[size] : sizeStylesVertical[size]}
          max-w-full max-h-full
          bg-white
          shadow-2xl
          ${animationStyles[position].enter}
          flex flex-col
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between p-4 border-b border-border-light flex-shrink-0">
            {title && (
              <h2
                id="drawer-title"
                className="font-heading text-heading-md text-navy-900"
              >
                {title}
              </h2>
            )}
            
            {showCloseButton && (
              <button
                type="button"
                onClick={onClose}
                className="
                  p-1 -m-1 ml-auto
                  text-neutral-400
                  hover:text-navy-900
                  transition-colors duration-fast
                  rounded-sm
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500
                "
                aria-label="Close drawer"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4">
          {children}
        </div>
      </div>
    </div>
  );

  // Render in portal
  if (typeof window === 'undefined') return null;
  
  const container = document.getElementById('modal-container') || document.body;
  return createPortal(drawerContent, container);
};

// =============================================================================
// DRAWER FOOTER
// =============================================================================

export interface DrawerFooterProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const DrawerFooter: React.FC<DrawerFooterProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        flex items-center justify-end gap-3
        p-4
        border-t border-border-light
        flex-shrink-0
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
};

export default Drawer;
