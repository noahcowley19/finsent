import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

export function formatPercent(value: number | null | undefined, includeSign: boolean = false): string {
  if (value === null || value === undefined) return 'N/A';
  const formatted = `${value.toFixed(2)}%`;
  return includeSign && value > 0 ? `+${formatted}` : formatted;
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  return new Intl.NumberFormat('en-US').format(value);
}

export function formatLargeNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  const absValue = Math.abs(value);
  
  if (absValue >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (absValue >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (absValue >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (absValue >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
  
  return formatCurrency(value);
}

export function formatCompactNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  const absValue = Math.abs(value);
  
  if (absValue >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (absValue >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (absValue >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  
  return value.toFixed(0);
}

export function getRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 30) return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

export function debounce<T extends (...args: unknown[]) => void>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function getStatusColor(status: 'positive' | 'negative' | 'neutral' | 'warning'): string {
  const colors = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    neutral: 'var(--neutral)',
    warning: 'var(--warning)',
  };
  return colors[status] || colors.neutral;
}

export function getStatusBgColor(status: 'positive' | 'negative' | 'neutral' | 'warning'): string {
  const colors = {
    positive: 'var(--positive-light)',
    negative: 'var(--negative-light)',
    neutral: 'var(--neutral-light)',
    warning: 'var(--warning-light)',
  };
  return colors[status] || colors.neutral;
}
