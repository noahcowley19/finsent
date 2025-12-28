'use client';

import React, { useEffect, useRef, ReactNode, HTMLAttributes } from 'react';
import { createPortal } from 'react-dom';

// =============================================================================
// TYPES
// =============================================================================

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface ModalProps {
  /** Whether modal is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Modal size */
  size?: ModalSize;
  /** Close on overlay click */
  closeOnOverlayClick?: boolean;
  /** Close on escape key */
  closeOnEscape?: boolean;
  /** Modal content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-2xl',
  full: 'max-w-[calc(100vw-2rem)] max-h-[calc(100vh-2rem)]',
};

// =============================================================================
// MODAL COMPONENT
// =============================================================================

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  size = 'md',
  closeOnOverlayClick = true,
  closeOnEscape = true,
  children,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && closeOnEscape) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, closeOnEscape, onClose]);

  // Focus trap (basic)
  useEffect(() => {
    if (isOpen && modalRef.current) {
      modalRef.current.focus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const modal = (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-ink-950/40 backdrop-blur-sm z-modal-backdrop animate-fade-in"
        onClick={closeOnOverlayClick ? onClose : undefined}
        aria-hidden="true"
      />

      {/* Modal Container */}
      <div className="fixed inset-0 z-modal flex items-center justify-center p-4">
        <div
          ref={modalRef}
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          className={`
            w-full ${sizeStyles[size]}
            bg-white rounded-2xl shadow-2xl
            animate-scale-in
            focus:outline-none
          `}
        >
          {children}
        </div>
      </div>
    </>
  );

  // Render to portal
  if (typeof document !== 'undefined') {
    return createPortal(modal, document.body);
  }

  return null;
};

// =============================================================================
// MODAL HEADER
// =============================================================================

export interface ModalHeaderProps extends HTMLAttributes<HTMLDivElement> {
  /** Title */
  title?: string;
  /** Subtitle */
  subtitle?: string;
  /** Show close button */
  showClose?: boolean;
  /** Close handler */
  onClose?: () => void;
  /** Custom content */
  children?: ReactNode;
}

export const ModalHeader: React.FC<ModalHeaderProps> = ({
  title,
  subtitle,
  showClose = true,
  onClose,
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`flex items-start justify-between p-6 border-b border-ink-100 ${className}`}
      {...props}
    >
      {children ? (
        children
      ) : (
        <div>
          {title && (
            <h2 className="text-heading-lg font-semibold text-ink-900 tracking-tight">
              {title}
            </h2>
          )}
          {subtitle && (
            <p className="mt-1 text-body-sm text-ink-500">{subtitle}</p>
          )}
        </div>
      )}
      {showClose && onClose && (
        <button
          onClick={onClose}
          className="p-2 -m-2 text-ink-400 hover:text-ink-600 transition-colors rounded-lg hover:bg-ink-50"
          aria-label="Close modal"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
};

// =============================================================================
// MODAL BODY
// =============================================================================

export interface ModalBodyProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const ModalBody: React.FC<ModalBodyProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={`p-6 ${className}`} {...props}>
      {children}
    </div>
  );
};

// =============================================================================
// MODAL FOOTER
// =============================================================================

export interface ModalFooterProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const ModalFooter: React.FC<ModalFooterProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`flex items-center justify-end gap-3 p-6 border-t border-ink-100 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default Modal;
