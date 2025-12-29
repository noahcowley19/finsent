'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';

// =============================================================================
// INTERACTIVE PRICE CHART - Professional Trading Chart
// =============================================================================
// Features: Candlesticks, Moving Averages, Patterns, Support/Resistance,
// Zoom, Fullscreen, Time Range Selector, Crosshair Tooltip
// =============================================================================

// -----------------------------------------------------------------------------
// TYPES
// -----------------------------------------------------------------------------

export interface OHLCV {
    date: Date;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface ChartConfig {
    showCandlesticks: boolean;
    showVolume: boolean;
    showMA20: boolean;
    showMA50: boolean;
    showMA200: boolean;
    showPatterns: boolean;
    showSupportResistance: boolean;
}

export type TimeRange = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | 'YTD' | 'ALL';

export interface InteractivePriceChartProps {
    symbol: string;
    name?: string;
    data?: OHLCV[];
}

// -----------------------------------------------------------------------------
// SAMPLE DATA GENERATOR
// -----------------------------------------------------------------------------

const generateSampleData = (days: number = 365, basePrice: number = 150): OHLCV[] => {
    const data: OHLCV[] = [];
    let price = basePrice;
    const now = new Date();

    for (let i = days; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);

        const volatility = 0.02;
        const change = (Math.random() - 0.48) * volatility * price;
        const open = price;
        const close = price + change;
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        const volume = Math.floor(Math.random() * 50000000) + 10000000;

        data.push({ date, open, high, low, close, volume });
        price = close;
    }

    return data;
};

// Calculate Simple Moving Average
const calculateSMA = (data: OHLCV[], period: number): (number | null)[] => {
    return data.map((_, index) => {
        if (index < period - 1) return null;
        const slice = data.slice(index - period + 1, index + 1);
        const sum = slice.reduce((acc, d) => acc + d.close, 0);
        return sum / period;
    });
};

// Find Support/Resistance levels
const findSupportResistance = (data: OHLCV[]): { support: number[]; resistance: number[] } => {
    if (data.length === 0) return { support: [], resistance: [] };

    const prices = data.map(d => d.close);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min;

    // Simple approach: divide into zones
    const zones = 5;
    const support: number[] = [];
    const resistance: number[] = [];

    for (let i = 1; i < zones; i++) {
        const level = min + (range * i / zones);
        if (i <= 2) support.push(level);
        else resistance.push(level);
    }

    return { support, resistance };
};

// -----------------------------------------------------------------------------
// CHART CONTROLS COMPONENT
// -----------------------------------------------------------------------------

interface ChartControlsProps {
    config: ChartConfig;
    onConfigChange: (key: keyof ChartConfig, value: boolean) => void;
    timeRange: TimeRange;
    onTimeRangeChange: (range: TimeRange) => void;
    onFullscreen: () => void;
    onZoomIn: () => void;
    onZoomOut: () => void;
    onResetZoom: () => void;
}

const ChartControls: React.FC<ChartControlsProps> = ({
    config,
    onConfigChange,
    timeRange,
    onTimeRangeChange,
    onFullscreen,
    onZoomIn,
    onZoomOut,
    onResetZoom,
}) => {
    const timeRanges: TimeRange[] = ['1D', '1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL'];

    const ToggleButton: React.FC<{ label: string; active: boolean; onClick: () => void }> = ({ label, active, onClick }) => (
        <button
            onClick={onClick}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${active
                    ? 'bg-electric-500 text-white shadow-sm'
                    : 'bg-cream-100 text-obsidian-600 hover:bg-cream-200'
                }`}
        >
            {label}
        </button>
    );

    return (
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
            {/* Time Range */}
            <div className="flex items-center gap-1 p-1 bg-cream-100/80 rounded-xl">
                {timeRanges.map((range) => (
                    <button
                        key={range}
                        onClick={() => onTimeRangeChange(range)}
                        className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${timeRange === range
                                ? 'bg-white text-obsidian-900 shadow-sm'
                                : 'text-obsidian-500 hover:text-obsidian-700'
                            }`}
                    >
                        {range}
                    </button>
                ))}
            </div>

            {/* Chart Type & Indicators */}
            <div className="flex items-center gap-2 flex-wrap">
                <ToggleButton
                    label="Candles"
                    active={config.showCandlesticks}
                    onClick={() => onConfigChange('showCandlesticks', !config.showCandlesticks)}
                />
                <ToggleButton
                    label="Volume"
                    active={config.showVolume}
                    onClick={() => onConfigChange('showVolume', !config.showVolume)}
                />
                <div className="w-px h-6 bg-cream-300" />
                <ToggleButton
                    label="MA20"
                    active={config.showMA20}
                    onClick={() => onConfigChange('showMA20', !config.showMA20)}
                />
                <ToggleButton
                    label="MA50"
                    active={config.showMA50}
                    onClick={() => onConfigChange('showMA50', !config.showMA50)}
                />
                <ToggleButton
                    label="MA200"
                    active={config.showMA200}
                    onClick={() => onConfigChange('showMA200', !config.showMA200)}
                />
                <div className="w-px h-6 bg-cream-300" />
                <ToggleButton
                    label="S/R"
                    active={config.showSupportResistance}
                    onClick={() => onConfigChange('showSupportResistance', !config.showSupportResistance)}
                />
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center gap-1">
                <button
                    onClick={onZoomOut}
                    className="p-2 rounded-lg bg-cream-100 text-obsidian-600 hover:bg-cream-200 transition-colors"
                    title="Zoom Out"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                    </svg>
                </button>
                <button
                    onClick={onResetZoom}
                    className="px-2 py-1.5 text-xs font-medium rounded-lg bg-cream-100 text-obsidian-600 hover:bg-cream-200 transition-colors"
                >
                    Reset
                </button>
                <button
                    onClick={onZoomIn}
                    className="p-2 rounded-lg bg-cream-100 text-obsidian-600 hover:bg-cream-200 transition-colors"
                    title="Zoom In"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                </button>
                <button
                    onClick={onFullscreen}
                    className="p-2 rounded-lg bg-cream-100 text-obsidian-600 hover:bg-cream-200 transition-colors ml-2"
                    title="Fullscreen"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

// -----------------------------------------------------------------------------
// PRICE DISPLAY COMPONENT
// -----------------------------------------------------------------------------

interface PriceDisplayProps {
    symbol: string;
    name?: string;
    currentPrice: number;
    change: number;
    changePercent: number;
    high: number;
    low: number;
    volume: number;
}

const PriceDisplay: React.FC<PriceDisplayProps> = ({
    symbol,
    name,
    currentPrice,
    change,
    changePercent,
    high,
    low,
    volume,
}) => {
    const isPositive = change >= 0;

    return (
        <div className="flex items-start justify-between mb-6">
            <div>
                <div className="flex items-center gap-3 mb-1">
                    <h2 className="text-2xl font-bold text-obsidian-900">{symbol}</h2>
                    {name && <span className="text-obsidian-500">{name}</span>}
                </div>
                <div className="flex items-baseline gap-3">
                    <span className="text-4xl font-bold text-obsidian-900">
                        ${currentPrice.toFixed(2)}
                    </span>
                    <span className={`text-lg font-semibold ${isPositive ? 'text-success-600' : 'text-coral-600'}`}>
                        {isPositive ? '+' : ''}{change.toFixed(2)} ({isPositive ? '+' : ''}{changePercent.toFixed(2)}%)
                    </span>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-6 text-right">
                <div>
                    <p className="text-xs text-obsidian-400 mb-0.5">High</p>
                    <p className="text-sm font-semibold text-obsidian-900">${high.toFixed(2)}</p>
                </div>
                <div>
                    <p className="text-xs text-obsidian-400 mb-0.5">Low</p>
                    <p className="text-sm font-semibold text-obsidian-900">${low.toFixed(2)}</p>
                </div>
                <div>
                    <p className="text-xs text-obsidian-400 mb-0.5">Volume</p>
                    <p className="text-sm font-semibold text-obsidian-900">
                        {(volume / 1000000).toFixed(1)}M
                    </p>
                </div>
            </div>
        </div>
    );
};

// -----------------------------------------------------------------------------
// MAIN CHART COMPONENT
// -----------------------------------------------------------------------------

export const InteractivePriceChart: React.FC<InteractivePriceChartProps> = ({
    symbol,
    name,
    data: providedData,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [zoom, setZoom] = useState(1);
    const [hoveredPoint, setHoveredPoint] = useState<OHLCV | null>(null);
    const [mouseX, setMouseX] = useState<number | null>(null);

    const [config, setConfig] = useState<ChartConfig>({
        showCandlesticks: true,
        showVolume: true,
        showMA20: true,
        showMA50: false,
        showMA200: false,
        showPatterns: false,
        showSupportResistance: false,
    });

    const [timeRange, setTimeRange] = useState<TimeRange>('6M');

    // Generate or use provided data
    const fullData = useMemo(() => providedData || generateSampleData(365, 150), [providedData]);

    // Filter data based on time range
    const data = useMemo(() => {
        const now = new Date();
        let daysBack = 180;

        switch (timeRange) {
            case '1D': daysBack = 1; break;
            case '1W': daysBack = 7; break;
            case '1M': daysBack = 30; break;
            case '3M': daysBack = 90; break;
            case '6M': daysBack = 180; break;
            case '1Y': daysBack = 365; break;
            case 'YTD':
                const startOfYear = new Date(now.getFullYear(), 0, 1);
                daysBack = Math.floor((now.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24));
                break;
            case 'ALL': daysBack = fullData.length; break;
        }

        return fullData.slice(-Math.min(daysBack, fullData.length));
    }, [fullData, timeRange]);

    // Calculate indicators
    const ma20 = useMemo(() => calculateSMA(data, 20), [data]);
    const ma50 = useMemo(() => calculateSMA(data, 50), [data]);
    const ma200 = useMemo(() => calculateSMA(data, 200), [data]);
    const supportResistance = useMemo(() => findSupportResistance(data), [data]);

    // Current price info
    const currentData = data[data.length - 1];
    const previousData = data[data.length - 2];
    const change = currentData ? currentData.close - (previousData?.close || currentData.open) : 0;
    const changePercent = previousData ? (change / previousData.close) * 100 : 0;

    // Config change handler
    const handleConfigChange = useCallback((key: keyof ChartConfig, value: boolean) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    }, []);

    // Zoom handlers
    const handleZoomIn = useCallback(() => setZoom(prev => Math.min(prev * 1.2, 5)), []);
    const handleZoomOut = useCallback(() => setZoom(prev => Math.max(prev / 1.2, 0.5)), []);
    const handleResetZoom = useCallback(() => setZoom(1), []);

    // Fullscreen handler
    const handleFullscreen = useCallback(() => {
        if (!isFullscreen && containerRef.current) {
            containerRef.current.requestFullscreen?.();
            setIsFullscreen(true);
        } else if (document.fullscreenElement) {
            document.exitFullscreen?.();
            setIsFullscreen(false);
        }
    }, [isFullscreen]);

    // Listen for fullscreen changes
    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }, []);

    // Draw chart
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || data.length === 0) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width;
        const height = rect.height;
        const volumeHeight = config.showVolume ? height * 0.15 : 0;
        const chartHeight = height - volumeHeight - 30;
        const padding = { top: 20, right: 60, bottom: 30, left: 10 };

        // Clear
        ctx.clearRect(0, 0, width, height);

        // Calculate price range
        const prices = data.flatMap(d => [d.high, d.low]);
        const minPrice = Math.min(...prices) * 0.995;
        const maxPrice = Math.max(...prices) * 1.005;
        const priceRange = maxPrice - minPrice;

        // Calculate volume range
        const maxVolume = Math.max(...data.map(d => d.volume));

        // Helper functions
        const xScale = (index: number) => padding.left + (index / (data.length - 1)) * (width - padding.left - padding.right) * zoom;
        const yScale = (price: number) => padding.top + ((maxPrice - price) / priceRange) * chartHeight;
        const volumeYScale = (vol: number) => height - padding.bottom - (vol / maxVolume) * volumeHeight;

        // Draw grid
        ctx.strokeStyle = '#E8E4DC';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 5; i++) {
            const y = padding.top + (i / 5) * chartHeight;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();

            // Price labels
            const price = maxPrice - (i / 5) * priceRange;
            ctx.fillStyle = '#6B7280';
            ctx.font = '11px Inter, system-ui, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`$${price.toFixed(2)}`, width - padding.right + 5, y + 4);
        }

        // Draw Support/Resistance
        if (config.showSupportResistance) {
            supportResistance.support.forEach(level => {
                const y = yScale(level);
                ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
            });

            supportResistance.resistance.forEach(level => {
                const y = yScale(level);
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
            });
        }

        // Draw candlesticks or line
        const candleWidth = Math.max(2, ((width - padding.left - padding.right) / data.length) * zoom * 0.7);

        if (config.showCandlesticks) {
            data.forEach((d, i) => {
                const x = xScale(i);
                const isUp = d.close >= d.open;

                // Wick
                ctx.strokeStyle = isUp ? '#22C55E' : '#EF4444';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, yScale(d.high));
                ctx.lineTo(x, yScale(d.low));
                ctx.stroke();

                // Body
                ctx.fillStyle = isUp ? '#22C55E' : '#EF4444';
                const bodyTop = yScale(Math.max(d.open, d.close));
                const bodyHeight = Math.max(1, Math.abs(yScale(d.open) - yScale(d.close)));
                ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
            });
        } else {
            // Line chart with gradient
            ctx.beginPath();
            data.forEach((d, i) => {
                const x = xScale(i);
                const y = yScale(d.close);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            const gradient = ctx.createLinearGradient(0, padding.top, 0, chartHeight);
            gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
            gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

            // Fill area
            ctx.lineTo(xScale(data.length - 1), chartHeight + padding.top);
            ctx.lineTo(xScale(0), chartHeight + padding.top);
            ctx.closePath();
            ctx.fillStyle = gradient;
            ctx.fill();

            // Draw line
            ctx.beginPath();
            data.forEach((d, i) => {
                const x = xScale(i);
                const y = yScale(d.close);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw Moving Averages
        const drawMA = (maData: (number | null)[], color: string) => {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            let started = false;
            maData.forEach((val, i) => {
                if (val === null) return;
                const x = xScale(i);
                const y = yScale(val);
                if (!started) {
                    ctx.moveTo(x, y);
                    started = true;
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();
        };

        if (config.showMA20) drawMA(ma20, '#F59E0B');
        if (config.showMA50) drawMA(ma50, '#8B5CF6');
        if (config.showMA200) drawMA(ma200, '#06B6D4');

        // Draw Volume
        if (config.showVolume) {
            data.forEach((d, i) => {
                const x = xScale(i);
                const barHeight = height - padding.bottom - volumeYScale(d.volume);
                const isUp = d.close >= d.open;
                ctx.fillStyle = isUp ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)';
                ctx.fillRect(x - candleWidth / 2, volumeYScale(d.volume), candleWidth, barHeight);
            });
        }

        // Draw crosshair
        if (mouseX !== null) {
            const chartX = mouseX - (canvas.getBoundingClientRect().left);
            ctx.strokeStyle = 'rgba(107, 114, 128, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(chartX, padding.top);
            ctx.lineTo(chartX, height - padding.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
        }

    }, [data, config, zoom, mouseX, ma20, ma50, ma200, supportResistance]);

    // Mouse move handler for crosshair
    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX;
        setMouseX(x);

        // Find nearest data point
        const chartWidth = rect.width - 70;
        const relX = e.clientX - rect.left - 10;
        const index = Math.round((relX / chartWidth) * (data.length - 1));

        if (index >= 0 && index < data.length) {
            setHoveredPoint(data[index]);
        }
    }, [data]);

    const handleMouseLeave = useCallback(() => {
        setMouseX(null);
        setHoveredPoint(null);
    }, []);

    if (data.length === 0) return null;

    return (
        <div
            ref={containerRef}
            className={`bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-6 transition-all ${isFullscreen ? 'fixed inset-4 z-50 overflow-auto' : ''
                }`}
        >
            <PriceDisplay
                symbol={symbol}
                name={name}
                currentPrice={currentData.close}
                change={change}
                changePercent={changePercent}
                high={currentData.high}
                low={currentData.low}
                volume={currentData.volume}
            />

            <ChartControls
                config={config}
                onConfigChange={handleConfigChange}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
                onFullscreen={handleFullscreen}
                onZoomIn={handleZoomIn}
                onZoomOut={handleZoomOut}
                onResetZoom={handleResetZoom}
            />

            {/* Chart Canvas */}
            <div className="relative">
                <canvas
                    ref={canvasRef}
                    className={`w-full ${isFullscreen ? 'h-[calc(100vh-300px)]' : 'h-[400px]'} cursor-crosshair`}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                />

                {/* Crosshair Tooltip */}
                {hoveredPoint && mouseX && (
                    <div
                        className="absolute top-2 left-2 bg-obsidian-900/90 backdrop-blur-sm text-white rounded-lg p-3 text-sm pointer-events-none z-10"
                    >
                        <p className="font-medium mb-1">
                            {hoveredPoint.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        </p>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            <span className="text-obsidian-300">Open:</span>
                            <span>${hoveredPoint.open.toFixed(2)}</span>
                            <span className="text-obsidian-300">High:</span>
                            <span>${hoveredPoint.high.toFixed(2)}</span>
                            <span className="text-obsidian-300">Low:</span>
                            <span>${hoveredPoint.low.toFixed(2)}</span>
                            <span className="text-obsidian-300">Close:</span>
                            <span className={hoveredPoint.close >= hoveredPoint.open ? 'text-success-400' : 'text-coral-400'}>
                                ${hoveredPoint.close.toFixed(2)}
                            </span>
                            <span className="text-obsidian-300">Volume:</span>
                            <span>{(hoveredPoint.volume / 1000000).toFixed(1)}M</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 mt-4 text-xs text-obsidian-500">
                {config.showMA20 && (
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 bg-amber-500 rounded" />
                        <span>MA20</span>
                    </div>
                )}
                {config.showMA50 && (
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 bg-purple-500 rounded" />
                        <span>MA50</span>
                    </div>
                )}
                {config.showMA200 && (
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 bg-cyan-500 rounded" />
                        <span>MA200</span>
                    </div>
                )}
                {config.showSupportResistance && (
                    <>
                        <div className="flex items-center gap-1.5">
                            <div className="w-3 h-0.5 bg-success-500 rounded" style={{ borderStyle: 'dashed' }} />
                            <span>Support</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-3 h-0.5 bg-coral-500 rounded" style={{ borderStyle: 'dashed' }} />
                            <span>Resistance</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default InteractivePriceChart;
