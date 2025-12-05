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
