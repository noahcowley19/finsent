'use client';

import React, {
  createContext,
  useContext,
  useCallback,
  useState,
  ReactNode,
  HTMLAttributes,
} from 'react';
import { createPortal } from 'react-dom';

// =============================================================================
// TYPES
// =============================================================================

export type ToastVariant = 'success' | 'error' | 'warning' | 'info';
export type ToastPosition = 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';

export interface Toast {
  id: string;
  variant: ToastVariant;
  title?: string;
  description: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface ToastProps extends HTMLAttributes<HTMLDivElement> {
  toast: Toast;
  onClose: (id: string) => void;
}

export interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  removeToast: (id: string) => void;
  clearToasts: () => void;
}

// =============================================================================
// CONTEXT
// =============================================================================

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

// =============================================================================
// STYLES
// =============================================================================

const variantStyles: Record<ToastVariant, { bg: string; icon: string; iconBg: string }> = {
  success: {
    bg: 'bg-white border-l-4 border-l-success-500',
    icon: 'text-success-500',
    iconBg: 'bg-success-50',
  },
  error: {
    bg: 'bg-white border-l-4 border-l-error-500',
    icon: 'text-error-500',
    iconBg: 'bg-error-50',
  },
  warning: {
    bg: 'bg-white border-l-4 border-l-warning-500',
    icon: 'text-warning-500',
    iconBg: 'bg-warning-50',
  },
  info: {
    bg: 'bg-white border-l-4 border-l-navy-500',
    icon: 'text-navy-500',
    iconBg: 'bg-navy-50',
  },
};

const icons: Record<ToastVariant, ReactNode> = {
  success: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  warning: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  info: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
};

// =============================================================================
// TOAST ITEM COMPONENT
// =============================================================================

const ToastItem: React.FC<ToastProps> = ({ toast, onClose, className = '', ...props }) => {
  const { variant, title, description, action } = toast;
  const styles = variantStyles[variant];

  return (
    <div
      role="alert"
      className={`
        ${styles.bg}
        w-full max-w-sm
        rounded-lg
        shadow-lg
        p-4
        animate-toast-enter
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      <div className="flex gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full ${styles.iconBg} ${styles.icon} flex items-center justify-center`}>
          {icons[variant]}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {title && (
            <p className="font-heading font-medium text-navy-900 text-body-sm">
              {title}
            </p>
          )}
          <p className={`text-body-sm text-neutral-600 ${title ? 'mt-0.5' : ''}`}>
            {description}
          </p>
          
          {action && (
            <button
              type="button"
              onClick={action.onClick}
              className="mt-2 text-body-sm font-medium text-navy-500 hover:text-navy-700 transition-colors"
            >
              {action.label}
            </button>
          )}
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={() => onClose(toast.id)}
          className="flex-shrink-0 p-1 -m-1 text-neutral-400 hover:text-navy-900 transition-colors"
          aria-label="Dismiss"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};

// =============================================================================
// TOAST PROVIDER
// =============================================================================

export interface ToastProviderProps {
  children: ReactNode;
  position?: ToastPosition;
  maxToasts?: number;
}

export const ToastProvider: React.FC<ToastProviderProps> = ({
  children,
  position = 'top-right',
  maxToasts = 5,
}) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback(
    (toast: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).substr(2, 9);
      const newToast: Toast = { ...toast, id };

      setToasts((prev) => {
        const updated = [newToast, ...prev].slice(0, maxToasts);
        return updated;
      });

      // Auto dismiss
      const duration = toast.duration ?? 5000;
      if (duration > 0) {
        setTimeout(() => {
          setToasts((prev) => prev.filter((t) => t.id !== id));
        }, duration);
      }

      return id;
    },
    [maxToasts]
  );

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const clearToasts = useCallback(() => {
    setToasts([]);
  }, []);

  const positionStyles: Record<ToastPosition, string> = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
  };

  const toastContainer =
    typeof window !== 'undefined' ? (
      createPortal(
        <div
          className={`fixed z-toast ${positionStyles[position]} flex flex-col gap-3 pointer-events-none`}
          aria-live="polite"
          aria-atomic="true"
        >
          {toasts.map((toast) => (
            <div key={toast.id} className="pointer-events-auto">
              <ToastItem toast={toast} onClose={removeToast} />
            </div>
          ))}
        </div>,
        document.getElementById('toast-container') || document.body
      )
    ) : null;

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, clearToasts }}>
      {children}
      {toastContainer}
    </ToastContext.Provider>
  );
};

// =============================================================================
// CONVENIENCE HOOK
// =============================================================================

export const useToastActions = () => {
  const { addToast } = useToast();

  return {
    success: (description: string, title?: string) =>
      addToast({ variant: 'success', title, description }),
    error: (description: string, title?: string) =>
      addToast({ variant: 'error', title, description }),
    warning: (description: string, title?: string) =>
      addToast({ variant: 'warning', title, description }),
    info: (description: string, title?: string) =>
      addToast({ variant: 'info', title, description }),
  };
};

export default ToastProvider;
