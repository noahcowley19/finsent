'use client';

// =============================================================================
// WATCHLIST PAGE
// =============================================================================
// Watchlist management page
//
// Location: frontend/app/watchlist/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { Section, AuthGuard } from '@/components/layout';
import { WatchlistTable, AddStockModal } from '@/components/watchlist';
import type { WatchlistItem } from '@/components/watchlist';

// Mock data - replace with real API calls
const mockWatchlist: WatchlistItem[] = [
  {
    id: '1',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 178.72,
    change: 2.34,
    changePercent: 1.33,
    sentiment: 42,
    addedAt: new Date('2024-12-01'),
  },
  {
    id: '2',
    symbol: 'TSLA',
    name: 'Tesla, Inc.',
    price: 248.50,
    change: -5.20,
    changePercent: -2.05,
    sentiment: -15,
    addedAt: new Date('2024-12-05'),
  },
  {
    id: '3',
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    price: 875.28,
    change: 12.45,
    changePercent: 1.44,
    sentiment: 65,
    addedAt: new Date('2024-12-10'),
  },
  {
    id: '4',
    symbol: 'AMD',
    name: 'Advanced Micro Devices',
    price: 164.50,
    change: -2.15,
    changePercent: -1.29,
    sentiment: 28,
    addedAt: new Date('2024-12-12'),
  },
  {
    id: '5',
    symbol: 'META',
    name: 'Meta Platforms, Inc.',
    price: 505.75,
    change: 8.32,
    changePercent: 1.67,
    sentiment: 35,
    addedAt: new Date('2024-12-15'),
  },
];

function WatchlistContent() {
  const [items, setItems] = useState<WatchlistItem[]>(mockWatchlist);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const handleRemove = (id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  };

  const handleAddToPortfolio = (item: WatchlistItem) => {
    console.log('Add to portfolio:', item);
    // TODO: Open add to portfolio modal
  };

  const handleAddStock = (symbol: string) => {
    // TODO: Fetch stock data and add to watchlist
    const newItem: WatchlistItem = {
      id: Date.now().toString(),
      symbol,
      name: `${symbol} Company`,
      price: Math.random() * 500 + 50,
      change: (Math.random() - 0.5) * 10,
      changePercent: (Math.random() - 0.5) * 5,
      sentiment: Math.round((Math.random() - 0.5) * 100),
      addedAt: new Date(),
    };
    setItems((prev) => [...prev, newItem]);
  };

  const existingSymbols = items.map((item) => item.symbol);

  return (
    <>
      {/* Header */}
      <Section spacing="md" background="gradient">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
              Watchlist
            </h1>
            <p className="text-body-md text-neutral-600">
              Track stocks you&apos;re interested in
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* View mode toggle */}
            <div className="flex bg-cream-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'list' ? 'bg-white shadow-sm text-navy-900' : 'text-neutral-500'
                }`}
                title="List view"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
              </button>
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'grid' ? 'bg-white shadow-sm text-navy-900' : 'text-neutral-500'
                }`}
                title="Grid view"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
              </button>
            </div>

            <button
              onClick={() => setIsAddModalOpen(true)}
              className="flex items-center gap-2 px-4 py-2.5 bg-terra-500 rounded-lg text-body-sm font-medium text-white hover:bg-terra-600 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Stock
            </button>
          </div>
        </div>
      </Section>

      {/* Stats */}
      <Section spacing="md" background="default">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl border border-border-light p-4">
            <p className="text-caption text-neutral-500 mb-1">Watching</p>
            <p className="text-heading-md font-semibold text-navy-900">{items.length} stocks</p>
          </div>
          <div className="bg-white rounded-xl border border-border-light p-4">
            <p className="text-caption text-neutral-500 mb-1">Bullish</p>
            <p className="text-heading-md font-semibold text-success-600">
              {items.filter((i) => (i.sentiment ?? 0) > 20).length}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-border-light p-4">
            <p className="text-caption text-neutral-500 mb-1">Bearish</p>
            <p className="text-heading-md font-semibold text-error-600">
              {items.filter((i) => (i.sentiment ?? 0) < -20).length}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-border-light p-4">
            <p className="text-caption text-neutral-500 mb-1">Avg Change</p>
            <p className={`text-heading-md font-semibold ${
              items.reduce((sum, i) => sum + i.changePercent, 0) / items.length >= 0
                ? 'text-success-600'
                : 'text-error-600'
            }`}>
              {items.length > 0
                ? `${(items.reduce((sum, i) => sum + i.changePercent, 0) / items.length).toFixed(2)}%`
                : '-'}
            </p>
          </div>
        </div>
      </Section>

      {/* Watchlist */}
      <Section spacing="lg" background="default">
        <WatchlistTable
          items={items}
          onRemove={handleRemove}
          onAddToPortfolio={handleAddToPortfolio}
          viewMode={viewMode}
        />
      </Section>

      {/* Add modal */}
      <AddStockModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdd={handleAddStock}
        existingSymbols={existingSymbols}
      />
    </>
  );
}

export default function WatchlistPage() {
  return (
    <AuthGuard>
      <WatchlistContent />
    </AuthGuard>
  );
}
