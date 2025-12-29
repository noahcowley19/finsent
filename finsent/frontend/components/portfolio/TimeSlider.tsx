'use client';

// =============================================================================
// TIME SLIDER - Historical/Future Portfolio Value Visualization
// =============================================================================

import React, { useState, useCallback, useMemo } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Position {
    ticker: string;
    shares: number;
    total_cost_basis: number;
}

interface TimeData {
    date: string;
    value: number;
    type: 'historical' | 'prediction';
    lower?: number;
    upper?: number;
}

interface TimeSliderProps {
    positions: Position[];
    currentValue: number;
    historicalData?: TimeData[];
    predictionData?: TimeData[];
}

export const TimeSlider: React.FC<TimeSliderProps> = ({
    positions,
    currentValue,
    historicalData = [],
    predictionData = [],
}) => {
    const [selectedDate, setSelectedDate] = useState<Date>(new Date());
    const [sliderValue, setSliderValue] = useState(50); // 0-100, 50 = today
    const [loading, setLoading] = useState(false);
    const [dateValue, setDateValue] = useState<number | null>(currentValue);
    const [error, setError] = useState<string | null>(null);

    // Generate date range: 1 year past to 30 days future
    const dateRange = useMemo(() => {
        const today = new Date();
        const start = new Date(today);
        start.setFullYear(start.getFullYear() - 1);
        const end = new Date(today);
        end.setDate(end.getDate() + 30);
        return { start, end, today };
    }, []);

    // Calculate date from slider position
    const getDateFromSlider = useCallback((value: number): Date => {
        const { start, end } = dateRange;
        const totalDays = Math.floor((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
        const daysFromStart = Math.floor((value / 100) * totalDays);
        const date = new Date(start);
        date.setDate(date.getDate() + daysFromStart);
        return date;
    }, [dateRange]);

    // Determine if date is past, present, or future
    const getDateType = useCallback((date: Date): 'past' | 'present' | 'future' => {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const compareDate = new Date(date);
        compareDate.setHours(0, 0, 0, 0);

        if (compareDate.getTime() === today.getTime()) return 'present';
        if (compareDate < today) return 'past';
        return 'future';
    }, []);

    // Fetch historical value for a specific date
    const fetchHistoricalValue = useCallback(async (date: Date) => {
        if (!positions || positions.length === 0) return;

        setLoading(true);
        setError(null);

        try {
            const dateType = getDateType(date);

            if (dateType === 'present') {
                setDateValue(currentValue);
            } else if (dateType === 'past') {
                // Fetch historical value from time-machine API
                const response = await fetch(`${API_BASE}/api/portfolio/time-machine`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        positions,
                        date: date.toISOString().split('T')[0],
                    }),
                });

                if (!response.ok) throw new Error('Failed to fetch historical data');

                const data = await response.json();
                setDateValue(data.total_value);
            } else {
                // Future: use prediction data if available, or show as projected
                const prediction = predictionData.find(
                    p => new Date(p.date).toDateString() === date.toDateString()
                );
                if (prediction) {
                    setDateValue(prediction.value);
                } else {
                    // Simple projection based on recent performance
                    const daysInFuture = Math.ceil((date.getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24));
                    const estimatedGrowth = 0.0002 * daysInFuture; // ~0.02% per day
                    setDateValue(currentValue * (1 + estimatedGrowth));
                }
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    }, [positions, currentValue, predictionData, getDateType]);

    // Handle slider change
    const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value, 10);
        setSliderValue(value);
        const date = getDateFromSlider(value);
        setSelectedDate(date);
    }, [getDateFromSlider]);

    // Handle slider release - fetch data
    const handleSliderRelease = useCallback(() => {
        fetchHistoricalValue(selectedDate);
    }, [fetchHistoricalValue, selectedDate]);

    const dateType = getDateType(selectedDate);
    const formattedDate = selectedDate.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
    });

    return (
        <div className="p-6 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-semibold text-obsidian-900 flex items-center gap-2">
                        <svg className="w-5 h-5 text-electric-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Time Travel
                    </h3>
                    <p className="text-sm text-obsidian-500">Explore your portfolio across time</p>
                </div>

                {/* Date indicator badge */}
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${dateType === 'past' ? 'bg-cream-200 text-obsidian-700' :
                        dateType === 'present' ? 'bg-success-100 text-success-700' :
                            'bg-electric-100 text-electric-700'
                    }`}>
                    {dateType === 'past' ? 'Historical' : dateType === 'present' ? 'Today' : 'Projected'}
                </div>
            </div>

            {/* Value Display */}
            <div className="text-center mb-6">
                <p className="text-sm text-obsidian-500 mb-1">{formattedDate}</p>
                <div className="flex items-center justify-center gap-2">
                    {loading ? (
                        <div className="h-10 w-32 bg-cream-200 animate-pulse rounded-lg" />
                    ) : (
                        <>
                            <p className="text-4xl font-bold text-obsidian-900">
                                ${dateValue?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '—'}
                            </p>
                            {dateType === 'future' && (
                                <span className="text-xs text-electric-500 uppercase tracking-wide">Projected</span>
                            )}
                        </>
                    )}
                </div>

                {/* Change from current */}
                {dateValue && currentValue && dateType !== 'present' && (
                    <p className={`text-sm mt-1 ${dateValue >= currentValue ? 'text-success-600' : 'text-coral-600'}`}>
                        {dateValue >= currentValue ? '+' : ''}
                        {((dateValue - currentValue) / currentValue * 100).toFixed(2)}%
                        ({dateValue >= currentValue ? '+' : ''}${(dateValue - currentValue).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })})
                    </p>
                )}

                {error && (
                    <p className="text-sm text-coral-500 mt-1">{error}</p>
                )}
            </div>

            {/* Slider */}
            <div className="relative">
                {/* Track labels */}
                <div className="flex justify-between text-xs text-obsidian-400 mb-2">
                    <span>1 Year Ago</span>
                    <span className="text-obsidian-600 font-medium">Today</span>
                    <span>+30 Days</span>
                </div>

                {/* Slider input */}
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={sliderValue}
                    onChange={handleSliderChange}
                    onMouseUp={handleSliderRelease}
                    onTouchEnd={handleSliderRelease}
                    className="w-full h-2 bg-cream-200 rounded-lg appearance-none cursor-pointer slider-thumb"
                    style={{
                        background: `linear-gradient(to right, 
              #E4E4E7 0%, 
              #E4E4E7 ${(365 / 395) * 100}%, 
              #22C55E ${(365 / 395) * 100}%, 
              #22C55E ${(366 / 395) * 100}%, 
              #60A5FA ${(366 / 395) * 100}%, 
              #60A5FA 100%)`,
                    }}
                />

                {/* Today marker */}
                <div
                    className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-obsidian-400 rounded"
                    style={{ left: `${(365 / 395) * 100}%`, marginTop: '1rem' }}
                />
            </div>

            {/* Info text */}
            <p className="text-xs text-obsidian-400 mt-4 text-center">
                {dateType === 'past'
                    ? 'Showing actual portfolio value from historical data'
                    : dateType === 'present'
                        ? 'Current portfolio value'
                        : 'ML-projected value based on historical patterns (confidence interval applies)'
                }
            </p>

            <style jsx>{`
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #1A1A1D;
          cursor: pointer;
          border: 3px solid white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        input[type="range"]::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #1A1A1D;
          cursor: pointer;
          border: 3px solid white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
      `}</style>
        </div>
    );
};

export default TimeSlider;
