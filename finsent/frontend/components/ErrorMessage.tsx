'use client';

import { ReactNode } from 'react';

interface ErrorMessageProps {
  message: string;
  title?: string;
  onRetry?: () => void;
  className?: string;
}

export default function ErrorMessage({ message, title = 'Error', onRetry, className = '' }: ErrorMessageProps) {
  return (
    <div
      className={className}
      style={{
        background: 'var(--negative-light)',
        border: '1px solid rgba(255, 107, 107, 0.3)',
        borderRadius: '12px',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
      }}
    >
      <div
        style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          background: 'var(--negative)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--negative)', marginBottom: '4px' }}>
          {title}
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          {message}
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            style={{
              marginTop: '12px',
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: 600,
              background: 'var(--negative)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
            }}
          >
            Try Again
          </button>
        )}
      </div>
    </div>
  );
}

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className = '' }: EmptyStateProps) {
  return (
    <div
      className={className}
      style={{
        textAlign: 'center',
        padding: '48px 24px',
      }}
    >
      {icon && (
        <div
          style={{
            width: '64px',
            height: '64px',
            margin: '0 auto 20px',
            borderRadius: '16px',
            background: 'var(--bg-secondary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            fontSize: '28px',
          }}
        >
          {icon}
        </div>
      )}
      <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '8px', color: 'var(--text-primary)' }}>
        {title}
      </h3>
      {description && (
        <p style={{ fontSize: '14px', color: 'var(--text-tertiary)', maxWidth: '360px', margin: '0 auto 20px' }}>
          {description}
        </p>
      )}
      {action && <div>{action}</div>}
    </div>
  );
}
