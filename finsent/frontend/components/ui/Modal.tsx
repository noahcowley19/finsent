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

// =============================================================================
// CONFIRM MODAL
// =============================================================================

export interface ConfirmModalProps extends Omit<ModalProps, 'children'> {
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  variant?: 'danger' | 'primary';
  isLoading?: boolean;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'primary',
  isLoading = false,
  ...props
}: ConfirmModalProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} size="sm" {...props}>
      <ModalHeader title={title} showClose={!isLoading} onClose={onClose} />
      <ModalBody>
        <p className="text-body-md text-ink-600">{description}</p>
      </ModalBody>
      <ModalFooter>
        <button
          onClick={onClose}
          disabled={isLoading}
          className="px-4 py-2 text-body-sm font-medium text-ink-600 hover:text-ink-900 transition-colors disabled:opacity-50"
        >
          {cancelLabel}
        </button>
        <button
          onClick={onConfirm}
          disabled={isLoading}
          className={`
            px-4 py-2 rounded-lg text-body-sm font-semibold text-white transition-all
            ${variant === 'danger' ? 'bg-error-600 hover:bg-error-700' : 'bg-ink-900 hover:bg-ink-800'}
            disabled:opacity-50
          `}
        >
          {isLoading ? (
            <svg className="w-4 h-4 animate-spin mx-auto" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            confirmLabel
          )}
        </button>
      </ModalFooter>
    </Modal>
  );
};

export default Modal;
