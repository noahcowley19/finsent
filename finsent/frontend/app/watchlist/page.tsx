'use client';

// =============================================================================
// WATCHLIST PAGE
// =============================================================================
// Watchlist management page
//
// Location: frontend/app/watchlist/page.tsx
//
// =============================================================================

import React, { useState, useEffect } from 'react';
import { Section, AuthGuard } from '@/components/layout';
import { WatchlistTable, AddStockModal } from '@/components/watchlist';
import type { WatchlistItem } from '@/components/watchlist';
import { useWatchlist, useQuickSearch } from '@/lib/hooks';
import { api } from '@/lib/api';

function WatchlistContent() {
  const { items: watchlistItems, enrichedItems, addItem, removeItem, refresh, loading: watchlistLoading } = useWatchlist();
  const { execute: quickSearch } = useQuickSearch();
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [items, setItems] = useState<WatchlistItem[]>([]);

  // Convert enriched items to WatchlistItem format
  useEffect(() => {
    const convertedItems: WatchlistItem[] = enrichedItems.map((item) => ({
      id: item.id,
      symbol: item.ticker,
      name: item.name || item.ticker,
      price: item.current_price || 0,
      change: item.change_dollar || 0,
      changePercent: item.change_percent || 0,
      sentiment: item.composite || 0,
      addedAt: new Date(item.dateAdded),
    }));
    setItems(convertedItems);
  }, [enrichedItems]);

  // Auto-refresh on mount and when items change
  useEffect(() => {
    if (watchlistItems.length > 0) {
      refresh();
    }
  }, [watchlistItems.length]);

  const handleRemove = (id: string) => {
    removeItem(id);
  };

  const handleAddToPortfolio = (item: WatchlistItem) => {
    console.log('Add to portfolio:', item);
    // TODO: Open add to portfolio modal
  };

  const handleAddStock = async (symbol: string) => {
    // Check if ticker is valid using quick search
    try {
      const result = await quickSearch(symbol.toUpperCase());
      
      if (result?.found) {
        // Add to watchlist
        addItem(symbol.toUpperCase());
        
        // Refresh to get enriched data
        setTimeout(() => refresh(), 100);
      } else {
        alert(`Stock ticker "${symbol}" not found. Please verify the symbol.`);
      }
    } catch (error) {
      console.error('Error adding stock:', error);
      alert('Failed to add stock. Please try again.');
    }
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
