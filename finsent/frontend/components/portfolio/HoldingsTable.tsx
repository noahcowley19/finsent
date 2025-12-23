'use client';

// =============================================================================
// HOLDINGS TABLE COMPONENT
// =============================================================================
// Displays portfolio holdings with details
//
// Location: frontend/components/portfolio/HoldingsTable.tsx
//
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';

export interface Holding {
  id: string;
  symbol: string;
  name: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
  value: number;
  gain: number;
  gainPercent: number;
  dayChange: number;
  dayChangePercent: number;
}

export interface HoldingsTableProps {
  holdings: Holding[];
  onEdit?: (holding: Holding) => void;
  onDelete?: (holdingId: string) => void;
  isLoading?: boolean;
}

type SortKey = 'symbol' | 'value' | 'gain' | 'dayChange';
type SortOrder = 'asc' | 'desc';

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

export const HoldingsTable: React.FC<HoldingsTableProps> = ({
  holdings,
  onEdit,
  onDelete,
  isLoading = false,
}) => {
  const [sortKey, setSortKey] = useState<SortKey>('value');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('desc');
    }
  };

  const sortedHoldings = [...holdings].sort((a, b) => {
    const aValue = a[sortKey];
    const bValue = b[sortKey];
    const multiplier = sortOrder === 'asc' ? 1 : -1;
    
    if (typeof aValue === 'string') {
      return aValue.localeCompare(bValue as string) * multiplier;
    }
    return ((aValue as number) - (bValue as number)) * multiplier;
  });

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => (
    <svg
      className={`w-4 h-4 ml-1 inline transition-transform ${
        sortKey === columnKey ? 'text-navy-600' : 'text-neutral-300'
      } ${sortKey === columnKey && sortOrder === 'asc' ? 'rotate-180' : ''}`}
      fill="currentColor"
      viewBox="0 0 20 20"
    >
      <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
    </svg>
  );

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light overflow-hidden">
        <div className="p-6 border-b border-border-light">
          <div className="w-32 h-6 rounded bg-cream-100 animate-pulse" />
        </div>
        <div className="animate-pulse">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center gap-4 p-4 border-b border-border-light">
              <div className="w-10 h-10 rounded-lg bg-cream-100" />
              <div className="flex-1">
                <div className="w-16 h-4 rounded bg-cream-100 mb-1" />
                <div className="w-24 h-3 rounded bg-cream-100" />
              </div>
              <div className="w-20 h-4 rounded bg-cream-100" />
              <div className="w-20 h-4 rounded bg-cream-100" />
              <div className="w-20 h-4 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (holdings.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-12 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <h3 className="font-heading font-semibold text-navy-900 mb-2">No holdings yet</h3>
        <p className="text-body-sm text-neutral-600 mb-4">
          Start building your portfolio by adding your first position.
        </p>
        <button className="px-4 py-2 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors">
          Add Position
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-border-light flex items-center justify-between">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Holdings ({holdings.length})
        </h3>
        <button className="px-4 py-2 bg-terra-500 text-white text-body-sm font-medium rounded-lg hover:bg-terra-600 transition-colors">
          + Add Position
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-cream-50">
              <th
                className="text-left py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider cursor-pointer hover:text-navy-700"
                onClick={() => handleSort('symbol')}
              >
                Stock <SortIcon columnKey="symbol" />
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Shares
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Avg Cost
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Price
              </th>
              <th
                className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider cursor-pointer hover:text-navy-700"
                onClick={() => handleSort('value')}
              >
                Value <SortIcon columnKey="value" />
              </th>
              <th
                className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider cursor-pointer hover:text-navy-700"
                onClick={() => handleSort('gain')}
              >
                Gain/Loss <SortIcon columnKey="gain" />
              </th>
              <th
                className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider cursor-pointer hover:text-navy-700"
                onClick={() => handleSort('dayChange')}
              >
                Today <SortIcon columnKey="dayChange" />
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-light">
            {sortedHoldings.map((holding) => (
              <tr key={holding.id} className="hover:bg-cream-50 transition-colors">
                <td className="py-4 px-4">
                  <Link href={`/stock/${holding.symbol}`} className="flex items-center gap-3 group">
                    <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                      <span className="text-body-sm font-bold text-navy-600">
                        {holding.symbol.slice(0, 2)}
                      </span>
                    </div>
                    <div>
                      <p className="text-body-sm font-medium text-navy-900 group-hover:text-navy-700">
                        {holding.symbol}
                      </p>
                      <p className="text-caption text-neutral-500 truncate max-w-[120px]">
                        {holding.name}
                      </p>
                    </div>
                  </Link>
                </td>
                <td className="py-4 px-4 text-body-sm text-navy-900 text-right tabular-nums">
                  {holding.shares.toLocaleString()}
                </td>
                <td className="py-4 px-4 text-body-sm text-navy-900 text-right tabular-nums">
                  {formatCurrency(holding.avgCost)}
                </td>
                <td className="py-4 px-4 text-body-sm text-navy-900 text-right tabular-nums">
                  {formatCurrency(holding.currentPrice)}
                </td>
                <td className="py-4 px-4 text-body-sm font-medium text-navy-900 text-right tabular-nums">
                  {formatCurrency(holding.value)}
                </td>
                <td className="py-4 px-4 text-right">
                  <span
                    className={`text-body-sm font-medium tabular-nums ${
                      holding.gain >= 0 ? 'text-success-600' : 'text-error-600'
                    }`}
                  >
                    {holding.gain >= 0 ? '+' : ''}{formatCurrency(holding.gain)}
                    <span className="text-caption ml-1">
                      ({holding.gain >= 0 ? '+' : ''}{holding.gainPercent.toFixed(2)}%)
                    </span>
                  </span>
                </td>
                <td className="py-4 px-4 text-right">
                  <span
                    className={`text-body-sm font-medium tabular-nums ${
                      holding.dayChange >= 0 ? 'text-success-600' : 'text-error-600'
                    }`}
                  >
                    {holding.dayChange >= 0 ? '+' : ''}{holding.dayChangePercent.toFixed(2)}%
                  </span>
                </td>
                <td className="py-4 px-4 text-right">
                  <div className="flex items-center justify-end gap-2">
                    {onEdit && (
                      <button
                        onClick={() => onEdit(holding)}
                        className="p-1.5 text-neutral-400 hover:text-navy-600 transition-colors"
                        title="Edit"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </button>
                    )}
                    {onDelete && (
                      <button
                        onClick={() => onDelete(holding.id)}
                        className="p-1.5 text-neutral-400 hover:text-error-600 transition-colors"
                        title="Delete"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default HoldingsTable;
