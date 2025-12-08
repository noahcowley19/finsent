'use client';

import { ReactNode } from 'react';

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (value: T[keyof T], row: T) => ReactNode;
  align?: 'left' | 'center' | 'right';
  width?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  emptyMessage?: string;
  className?: string;
  stickyHeader?: boolean;
  maxHeight?: string;
}

export default function DataTable<T extends Record<string, unknown>>({ 
  columns, 
  data, 
  onRowClick,
  emptyMessage = 'No data available',
  className = '',
  stickyHeader = false,
  maxHeight,
}: DataTableProps<T>) {
  const getValue = (row: T, key: string): unknown => {
    const keys = key.split('.');
    let value: unknown = row;
    for (const k of keys) {
      value = (value as Record<string, unknown>)?.[k];
    }
    return value;
  };

  return (
    <div 
      className={className}
      style={{ 
        overflowX: 'auto',
        maxHeight: maxHeight,
        overflowY: maxHeight ? 'auto' : undefined,
      }}
    >
      <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead style={{ position: stickyHeader ? 'sticky' : undefined, top: 0, zIndex: 10 }}>
          <tr>
            {columns.map((col, i) => (
              <th
                key={i}
                style={{
                  textAlign: col.align || 'left',
                  width: col.width,
                  padding: '12px 16px',
                  fontSize: '11px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: 'var(--text-muted)',
                  background: 'var(--bg-secondary)',
                  borderBottom: '1px solid var(--border)',
                  whiteSpace: 'nowrap',
                }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td 
                colSpan={columns.length}
                style={{
                  textAlign: 'center',
                  padding: '40px',
                  color: 'var(--text-muted)',
                  fontSize: '14px',
                }}
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                onClick={() => onRowClick?.(row)}
                style={{
                  cursor: onRowClick ? 'pointer' : 'default',
                  transition: 'background 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  if (onRowClick) e.currentTarget.style.background = 'var(--bg-elevated)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                {columns.map((col, colIndex) => {
                  const value = getValue(row, col.key as string);
                  return (
                    <td
                      key={colIndex}
                      style={{
                        textAlign: col.align || 'left',
                        padding: '14px 16px',
                        fontSize: '13px',
                        color: 'var(--text-primary)',
                        borderBottom: '1px solid var(--border)',
                      }}
                    >
                      {col.render ? col.render(value as T[keyof T], row) : String(value ?? '')}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

interface SimpleTableProps {
  rows: Array<{ label: string; value: ReactNode; highlight?: boolean }>;
  className?: string;
}

export function SimpleTable({ rows, className = '' }: SimpleTableProps) {
  return (
    <div className={className}>
      {rows.map((row, i) => (
        <div
          key={i}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 0',
            borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none',
          }}
        >
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{row.label}</span>
          <span
            style={{
              fontSize: '14px',
              fontWeight: row.highlight ? 700 : 600,
              fontFamily: "'JetBrains Mono', monospace",
              color: row.highlight ? 'var(--accent)' : 'var(--text-primary)',
            }}
          >
            {row.value}
          </span>
        </div>
      ))}
    </div>
  );
}
