'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Modal, ModalHeader, ModalBody } from '@/components/ui';
import { useQuickSearch } from '@/lib/hooks';

// =============================================================================
// ADD POSITION MODAL
// =============================================================================
// Modal for adding new positions to portfolio with ticker search,
// shares input, and cost basis.
// =============================================================================

export interface AddPositionModalProps {
    isOpen: boolean;
    onClose: () => void;
    onAddPosition: (position: {
        ticker: string;
        shares: number;
        cost_basis: number;
        name?: string;
    }) => void;
}

export const AddPositionModal: React.FC<AddPositionModalProps> = ({
    isOpen,
    onClose,
    onAddPosition,
}) => {
    const [ticker, setTicker] = useState('');
    const [shares, setShares] = useState('');
    const [costBasis, setCostBasis] = useState('');
    const [tickerError, setTickerError] = useState('');
    const [isValidating, setIsValidating] = useState(false);
    const [validatedTicker, setValidatedTicker] = useState<{ symbol: string; name: string } | null>(null);

    const { execute: quickSearch } = useQuickSearch();

    // Reset form when modal closes
    useEffect(() => {
        if (!isOpen) {
            setTicker('');
            setShares('');
            setCostBasis('');
            setTickerError('');
            setValidatedTicker(null);
        }
    }, [isOpen]);

    // Validate ticker on blur or after typing stops
    const validateTicker = useCallback(async () => {
        if (!ticker.trim()) {
            setTickerError('');
            setValidatedTicker(null);
            return;
        }

        setIsValidating(true);
        setTickerError('');

        try {
            const result = await quickSearch(ticker.toUpperCase());
            if (result?.found) {
                setValidatedTicker({ symbol: ticker.toUpperCase(), name: result.name || ticker.toUpperCase() });
                setTickerError('');
            } else {
                setValidatedTicker(null);
                setTickerError('Ticker not found');
            }
        } catch (error) {
            setTickerError('Error validating ticker');
            setValidatedTicker(null);
        } finally {
            setIsValidating(false);
        }
    }, [ticker, quickSearch]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!validatedTicker) {
            setTickerError('Please enter a valid ticker');
            return;
        }

        const sharesNum = parseFloat(shares);
        const costNum = parseFloat(costBasis);

        if (isNaN(sharesNum) || sharesNum <= 0) {
            return;
        }

        if (isNaN(costNum) || costNum <= 0) {
            return;
        }

        onAddPosition({
            ticker: validatedTicker.symbol,
            shares: sharesNum,
            cost_basis: costNum,
            name: validatedTicker.name,
        });

        onClose();
    };

    const isFormValid = validatedTicker && parseFloat(shares) > 0 && parseFloat(costBasis) > 0;

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="sm">
            <ModalHeader title="Add Position" onClose={onClose} />
            <ModalBody>
                <form onSubmit={handleSubmit} className="space-y-5">
                    {/* Ticker Input */}
                    <div>
                        <label className="block text-sm font-medium text-obsidian-700 mb-2">
                            Ticker Symbol
                        </label>
                        <div className="relative">
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => {
                                    setTicker(e.target.value.toUpperCase());
                                    setValidatedTicker(null);
                                }}
                                onBlur={validateTicker}
                                placeholder="e.g., AAPL"
                                className={`
                w-full h-11 px-4 rounded-xl border text-obsidian-900
                focus:outline-none focus:ring-2 transition-all
                ${tickerError
                                        ? 'border-coral-500 focus:ring-coral-500/20'
                                        : validatedTicker
                                            ? 'border-success-500 focus:ring-success-500/20'
                                            : 'border-cream-300 focus:ring-electric-500/20 focus:border-electric-500'
                                    }
              `}
                            />
                            {isValidating && (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                    <div className="w-5 h-5 border-2 border-electric-500 border-t-transparent rounded-full animate-spin" />
                                </div>
                            )}
                            {validatedTicker && !isValidating && (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-success-500">
                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                    </svg>
                                </div>
                            )}
                        </div>
                        {tickerError && (
                            <p className="mt-1.5 text-sm text-coral-500">{tickerError}</p>
                        )}
                        {validatedTicker && (
                            <p className="mt-1.5 text-sm text-success-600">{validatedTicker.name}</p>
                        )}
                    </div>

                    {/* Shares Input */}
                    <div>
                        <label className="block text-sm font-medium text-obsidian-700 mb-2">
                            Number of Shares
                        </label>
                        <input
                            type="number"
                            value={shares}
                            onChange={(e) => setShares(e.target.value)}
                            placeholder="e.g., 100"
                            min="0.01"
                            step="0.01"
                            className="w-full h-11 px-4 rounded-xl border border-cream-300 text-obsidian-900 focus:outline-none focus:ring-2 focus:ring-electric-500/20 focus:border-electric-500 transition-all"
                        />
                    </div>

                    {/* Cost Basis Input */}
                    <div>
                        <label className="block text-sm font-medium text-obsidian-700 mb-2">
                            Cost Basis (per share)
                        </label>
                        <div className="relative">
                            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-obsidian-400">$</span>
                            <input
                                type="number"
                                value={costBasis}
                                onChange={(e) => setCostBasis(e.target.value)}
                                placeholder="e.g., 150.00"
                                min="0.01"
                                step="0.01"
                                className="w-full h-11 pl-8 pr-4 rounded-xl border border-cream-300 text-obsidian-900 focus:outline-none focus:ring-2 focus:ring-electric-500/20 focus:border-electric-500 transition-all"
                            />
                        </div>
                    </div>

                    {/* Summary */}
                    {isFormValid && (
                        <div className="p-4 rounded-xl bg-cream-100 border border-cream-200">
                            <p className="text-sm text-obsidian-600">
                                Adding <strong>{parseFloat(shares).toLocaleString()}</strong> shares of{' '}
                                <strong>{validatedTicker?.symbol}</strong> at{' '}
                                <strong>${parseFloat(costBasis).toFixed(2)}</strong> per share
                            </p>
                            <p className="text-sm text-obsidian-500 mt-1">
                                Total cost: <strong>${(parseFloat(shares) * parseFloat(costBasis)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong>
                            </p>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-3 pt-2">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 h-11 rounded-xl border border-cream-300 text-obsidian-700 font-medium hover:bg-cream-100 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={!isFormValid}
                            className={`
              flex-1 h-11 rounded-xl font-medium transition-all
              ${isFormValid
                                    ? 'bg-obsidian-900 text-white hover:bg-obsidian-850'
                                    : 'bg-cream-200 text-obsidian-400 cursor-not-allowed'
                                }
            `}
                        >
                            Add Position
                        </button>
                    </div>
                </form>
            </ModalBody>
        </Modal>
    );
};

export default AddPositionModal;
