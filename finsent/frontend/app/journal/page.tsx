'use client';

import React, { useState, useEffect } from 'react';
import { Section } from '@/components/layout';

interface TradeEntry {
    id: string;
    date: string;
    ticker: string;
    action: 'buy' | 'sell';
    shares: number;
    price: number;
    notes: string;
    emotion: 'confident' | 'neutral' | 'fearful';
    strategy: string;
    outcome?: 'win' | 'loss' | 'pending';
    exitPrice?: number;
    exitDate?: string;
}

const STORAGE_KEY = 'finsent_journal';

export default function JournalPage() {
    const [entries, setEntries] = useState<TradeEntry[]>([]);
    const [showForm, setShowForm] = useState(false);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [filter, setFilter] = useState<'all' | 'win' | 'loss' | 'pending'>('all');

    const [form, setForm] = useState<Partial<TradeEntry>>({
        action: 'buy',
        emotion: 'neutral',
        strategy: '',
    });

    useEffect(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            setEntries(JSON.parse(stored));
        }
    }, []);

    const saveEntries = (newEntries: TradeEntry[]) => {
        setEntries(newEntries);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newEntries));
    };

    const handleSubmit = () => {
        if (!form.ticker || !form.shares || !form.price) return;

        const entry: TradeEntry = {
            id: editingId || Date.now().toString(),
            date: form.date || new Date().toISOString().split('T')[0],
            ticker: (form.ticker || '').toUpperCase(),
            action: form.action || 'buy',
            shares: Number(form.shares),
            price: Number(form.price),
            notes: form.notes || '',
            emotion: form.emotion || 'neutral',
            strategy: form.strategy || '',
            outcome: form.outcome || 'pending',
            exitPrice: form.exitPrice,
            exitDate: form.exitDate,
        };

        if (editingId) {
            saveEntries(entries.map(e => e.id === editingId ? entry : e));
        } else {
            saveEntries([entry, ...entries]);
        }

        setForm({ action: 'buy', emotion: 'neutral', strategy: '' });
        setShowForm(false);
        setEditingId(null);
    };

    const deleteTrade = (id: string) => {
        if (confirm('Delete this trade entry?')) {
            saveEntries(entries.filter(e => e.id !== id));
        }
    };

    const editTrade = (entry: TradeEntry) => {
        setForm(entry);
        setEditingId(entry.id);
        setShowForm(true);
    };

    const closeTrade = (id: string, exitPrice: number) => {
        const entry = entries.find(e => e.id === id);
        if (!entry) return;

        const outcome = entry.action === 'buy'
            ? (exitPrice > entry.price ? 'win' : 'loss')
            : (exitPrice < entry.price ? 'win' : 'loss');

        saveEntries(entries.map(e => e.id === id ? {
            ...e,
            outcome,
            exitPrice,
            exitDate: new Date().toISOString().split('T')[0],
        } : e));
    };

    const filteredEntries = entries.filter(e => filter === 'all' || e.outcome === filter);

    const stats = {
        total: entries.length,
        wins: entries.filter(e => e.outcome === 'win').length,
        losses: entries.filter(e => e.outcome === 'loss').length,
        pending: entries.filter(e => e.outcome === 'pending').length,
        winRate: entries.filter(e => e.outcome !== 'pending').length > 0
            ? (entries.filter(e => e.outcome === 'win').length / entries.filter(e => e.outcome !== 'pending').length * 100)
            : 0,
    };

    return (
        <>
            <Section spacing="md" background="gradient">
                <div className="flex justify-between items-start">
                    <div>
                        <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                            Investment Journal
                        </h1>
                        <p className="text-body-md text-neutral-600">
                            Track your trades, document your reasoning, and learn from outcomes.
                        </p>
                    </div>
                    <button
                        onClick={() => { setShowForm(true); setEditingId(null); setForm({ action: 'buy', emotion: 'neutral', strategy: '' }); }}
                        className="px-5 py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600"
                    >
                        + New Entry
                    </button>
                </div>
            </Section>

            {/* Stats */}
            <Section spacing="sm" background="alt">
                <div className="flex flex-wrap justify-center gap-8">
                    <div className="text-center">
                        <p className="text-display-xs font-bold text-navy-900">{stats.total}</p>
                        <p className="text-caption text-neutral-500">Total Trades</p>
                    </div>
                    <div className="text-center">
                        <p className="text-display-xs font-bold text-success-600">{stats.wins}</p>
                        <p className="text-caption text-neutral-500">Wins</p>
                    </div>
                    <div className="text-center">
                        <p className="text-display-xs font-bold text-error-600">{stats.losses}</p>
                        <p className="text-caption text-neutral-500">Losses</p>
                    </div>
                    <div className="text-center">
                        <p className="text-display-xs font-bold text-warning-600">{stats.pending}</p>
                        <p className="text-caption text-neutral-500">Open</p>
                    </div>
                    <div className="text-center">
                        <p className={`text-display-xs font-bold ${stats.winRate >= 50 ? 'text-success-600' : 'text-error-600'}`}>
                            {stats.winRate.toFixed(0)}%
                        </p>
                        <p className="text-caption text-neutral-500">Win Rate</p>
                    </div>
                </div>
            </Section>

            <Section spacing="md" background="default">
                {/* Entry Form Modal */}
                {showForm && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                        <div className="bg-white rounded-xl p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto">
                            <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                                {editingId ? 'Edit Trade' : 'Log New Trade'}
                            </h2>

                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-caption text-neutral-500 mb-1">Date</label>
                                        <input
                                            type="date"
                                            value={form.date || new Date().toISOString().split('T')[0]}
                                            onChange={(e) => setForm({ ...form, date: e.target.value })}
                                            className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-caption text-neutral-500 mb-1">Ticker</label>
                                        <input
                                            type="text"
                                            value={form.ticker || ''}
                                            onChange={(e) => setForm({ ...form, ticker: e.target.value.toUpperCase() })}
                                            placeholder="AAPL"
                                            className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                        />
                                    </div>
                                </div>

                                <div className="grid grid-cols-3 gap-4">
                                    <div>
                                        <label className="block text-caption text-neutral-500 mb-1">Action</label>
                                        <select
                                            value={form.action}
                                            onChange={(e) => setForm({ ...form, action: e.target.value as 'buy' | 'sell' })}
                                            className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                        >
                                            <option value="buy">Buy</option>
                                            <option value="sell">Sell</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-caption text-neutral-500 mb-1">Shares</label>
                                        <input
                                            type="number"
                                            value={form.shares || ''}
                                            onChange={(e) => setForm({ ...form, shares: Number(e.target.value) })}
                                            className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-caption text-neutral-500 mb-1">Price</label>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={form.price || ''}
                                            onChange={(e) => setForm({ ...form, price: Number(e.target.value) })}
                                            className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                        />
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Strategy/Thesis</label>
                                    <input
                                        type="text"
                                        value={form.strategy || ''}
                                        onChange={(e) => setForm({ ...form, strategy: e.target.value })}
                                        placeholder="e.g., Momentum breakout, Earnings play"
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                    />
                                </div>

                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Emotion/Confidence</label>
                                    <div className="flex gap-3">
                                        {(['confident', 'neutral', 'fearful'] as const).map(emotion => (
                                            <button
                                                key={emotion}
                                                type="button"
                                                onClick={() => setForm({ ...form, emotion })}
                                                className={`px-4 py-2 rounded-lg text-body-sm font-medium ${form.emotion === emotion
                                                    ? emotion === 'confident' ? 'bg-success-100 text-success-700' :
                                                        emotion === 'fearful' ? 'bg-error-100 text-error-700' :
                                                            'bg-neutral-100 text-neutral-700'
                                                    : 'bg-cream-100 text-neutral-500'
                                                    }`}
                                            >
                                                {emotion === 'confident' ? '😎' : emotion === 'fearful' ? '😰' : '😐'} {emotion}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Notes</label>
                                    <textarea
                                        value={form.notes || ''}
                                        onChange={(e) => setForm({ ...form, notes: e.target.value })}
                                        placeholder="Why are you making this trade? What's your thesis?"
                                        rows={3}
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg resize-none"
                                    />
                                </div>
                            </div>

                            <div className="flex gap-3 mt-6">
                                <button
                                    onClick={() => { setShowForm(false); setEditingId(null); }}
                                    className="flex-1 py-3 border border-border-medium rounded-lg font-medium hover:bg-cream-50"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleSubmit}
                                    className="flex-1 py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600"
                                >
                                    {editingId ? 'Update' : 'Save Entry'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Filter Tabs */}
                <div className="flex border-b border-border-light mb-6">
                    {(['all', 'pending', 'win', 'loss'] as const).map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`px-4 py-3 text-body-sm font-medium border-b-2 -mb-px transition-colors ${filter === f
                                ? 'border-terra-500 text-terra-600'
                                : 'border-transparent text-neutral-500 hover:text-neutral-700'
                                }`}
                        >
                            {f === 'all' ? 'All Trades' : f === 'pending' ? 'Open' : f === 'win' ? 'Wins' : 'Losses'}
                        </button>
                    ))}
                </div>

                {/* Entries */}
                {filteredEntries.length > 0 ? (
                    <div className="space-y-4">
                        {filteredEntries.map(entry => (
                            <div key={entry.id} className="bg-white rounded-xl border border-border-light p-5">
                                <div className="flex items-start justify-between">
                                    <div className="flex items-center gap-3">
                                        <span className={`px-3 py-1 rounded-full text-caption font-medium ${entry.action === 'buy' ? 'bg-success-100 text-success-700' : 'bg-error-100 text-error-700'
                                            }`}>
                                            {entry.action.toUpperCase()}
                                        </span>
                                        <span className="font-bold text-navy-900">{entry.ticker}</span>
                                        <span className="text-neutral-500">{entry.shares} shares @ ${entry.price.toFixed(2)}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className={`px-2 py-1 rounded text-caption font-medium ${entry.outcome === 'win' ? 'bg-success-100 text-success-700' :
                                            entry.outcome === 'loss' ? 'bg-error-100 text-error-700' :
                                                'bg-warning-100 text-warning-700'
                                            }`}>
                                            {entry.outcome === 'pending' ? 'Open' : entry.outcome}
                                        </span>
                                        <button onClick={() => editTrade(entry)} className="text-neutral-400 hover:text-neutral-600">✏️</button>
                                        <button onClick={() => deleteTrade(entry.id)} className="text-neutral-400 hover:text-error-600">🗑️</button>
                                    </div>
                                </div>

                                <div className="mt-3 text-body-sm text-neutral-600">
                                    <span className="text-caption text-neutral-400">{entry.date}</span>
                                    {entry.strategy && <span className="ml-3">📋 {entry.strategy}</span>}
                                </div>

                                {entry.notes && (
                                    <p className="mt-2 text-body-sm text-neutral-600 italic">"{entry.notes}"</p>
                                )}

                                {entry.outcome === 'pending' && (
                                    <div className="mt-3 flex items-center gap-2">
                                        <input
                                            type="number"
                                            step="0.01"
                                            placeholder="Exit price"
                                            className="w-28 px-2 py-1 border border-border-medium rounded text-body-sm"
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter') {
                                                    const price = parseFloat((e.target as HTMLInputElement).value);
                                                    if (price) closeTrade(entry.id, price);
                                                }
                                            }}
                                        />
                                        <span className="text-caption text-neutral-400">Press Enter to close</span>
                                    </div>
                                )}

                                {entry.exitPrice && (
                                    <div className="mt-2 text-body-sm">
                                        <span className="text-neutral-500">Closed @ ${entry.exitPrice.toFixed(2)}</span>
                                        <span className={`ml-2 font-medium ${entry.outcome === 'win' ? 'text-success-600' : 'text-error-600'
                                            }`}>
                                            {((entry.exitPrice - entry.price) / entry.price * 100 * (entry.action === 'buy' ? 1 : -1)).toFixed(1)}%
                                        </span>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-16 bg-cream-50 rounded-xl">
                        <p className="text-display-xs">📓</p>
                        <p className="text-neutral-600 font-medium mt-2">No trades logged yet</p>
                        <p className="text-neutral-500 text-body-sm">Click "New Entry" to start tracking</p>
                    </div>
                )}
            </Section>
        </>
    );
}
