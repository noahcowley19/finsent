import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export default function ErrorMessage({ message, onRetry, className }: ErrorMessageProps) {
  return (
    <div className={cn('error-message', className)}>
      <div className="flex items-center justify-center gap-3">
        <svg 
          className="w-5 h-5 flex-shrink-0" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
          />
        </svg>
        <span>{message}</span>
        {onRetry && (
          <button 
            onClick={onRetry}
            className="ml-2 underline hover:no-underline font-medium"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}

interface EmptyStateProps {
  title?: string;
  message: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ 
  title, 
  message, 
  icon, 
  action,
  className 
}: EmptyStateProps) {
  return (
    <div className={cn('text-center py-12', className)}>
      {icon && (
        <div className="mb-4 text-secondary">
          {icon}
        </div>
      )}
      {title && (
        <h3 className="text-lg font-semibold text-primary mb-2">{title}</h3>
      )}
      <p className="text-secondary mb-4">{message}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
