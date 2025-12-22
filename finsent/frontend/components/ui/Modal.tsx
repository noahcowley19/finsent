'use client';

import React, { useEffect, useCallback, ReactNode, HTMLAttributes } from 'react';
import { createPortal } from 'react-dom';

// =============================================================================
// TYPES
// =============================================================================

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface ModalProps extends HTMLAttributes<HTMLDivElement> {
  /** Whether modal is open */
  isOpen: boolean;
  /** Called when modal should close */
  onClose: () => void;
  /** Modal title */
  title?: string;
  /** Modal description */
  description?: string;
  /** Modal size */
  size?: ModalSize;
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

const sizeStyles: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-[calc(100vw-2rem)] max-h-[calc(100vh-2rem)]',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  description,
  size = 'md',
  closeOnBackdrop = true,
  closeOnEscape = true,
  showCloseButton = true,
  children,
  className = '',
  ...props
}) => {
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

  const modalContent = (
    <div
      className="fixed inset-0 z-modal flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'modal-title' : undefined}
      aria-describedby={description ? 'modal-description' : undefined}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-navy-900/40 backdrop-blur-sm animate-fade-in"
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Modal content */}
      <div
        className={`
          relative
          w-full
          ${sizeStyles[size]}
          bg-white
          rounded-xl
          shadow-2xl
          animate-modal-enter
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-start justify-between p-6 pb-0">
            <div>
              {title && (
                <h2
                  id="modal-title"
                  className="font-heading text-heading-lg text-navy-900"
                >
                  {title}
                </h2>
              )}
              {description && (
                <p
                  id="modal-description"
                  className="mt-1 text-body-sm text-neutral-600"
                >
                  {description}
                </p>
              )}
            </div>
            
            {showCloseButton && (
              <button
                type="button"
                onClick={onClose}
                className="
                  p-1 -m-1
                  text-neutral-400
                  hover:text-navy-900
                  transition-colors duration-fast
                  rounded-sm
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500
                "
                aria-label="Close modal"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Body */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );

  // Render in portal
  if (typeof window === 'undefined') return null;
  
  const container = document.getElementById('modal-container') || document.body;
  return createPortal(modalContent, container);
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
      className={`
        flex items-center justify-end gap-3
        pt-4 mt-4
        border-t border-border-light
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// CONFIRM MODAL
// =============================================================================

export interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'default' | 'danger';
  isLoading?: boolean;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title = 'Confirm',
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'default',
  isLoading = false,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
      closeOnBackdrop={!isLoading}
      closeOnEscape={!isLoading}
    >
      <p className="text-body-md text-neutral-600">{message}</p>
      
      <ModalFooter>
        <button
          type="button"
          onClick={onClose}
          disabled={isLoading}
          className="
            px-4 py-2
            text-body-md font-medium text-neutral-600
            hover:text-navy-900
            transition-colors duration-fast
            disabled:opacity-50
          "
        >
          {cancelText}
        </button>
        <button
          type="button"
          onClick={onConfirm}
          disabled={isLoading}
          className={`
            px-4 py-2
            text-body-md font-medium text-white
            rounded-sm
            transition-all duration-fast
            disabled:opacity-50
            ${variant === 'danger' 
              ? 'bg-error-500 hover:bg-error-600' 
              : 'bg-terra-500 hover:bg-terra-600'
            }
          `}
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Processing...
            </span>
          ) : (
            confirmText
          )}
        </button>
      </ModalFooter>
    </Modal>
  );
};

export default Modal;
