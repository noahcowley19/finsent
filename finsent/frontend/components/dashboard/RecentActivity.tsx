'use client';

import React from 'react';
import Link from 'next/link';

// =============================================================================
// TYPES
// =============================================================================

export interface Activity {
  id: string;
  type: 'search' | 'analysis' | 'watchlist' | 'portfolio';
  title: string;
  description?: string;
  timestamp: Date | string;
  link?: string;
}

export interface RecentActivityProps {
  activities: Activity[];
  maxItems?: number;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

const getActivityIcon = (type: Activity['type']) => {
  switch (type) {
    case 'search':
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      );
    case 'analysis':
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      );
    case 'watchlist':
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
      );
    case 'portfolio':
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      );
    default:
      return null;
  }
};

const formatTime = (timestamp: Date | string) => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
};

// =============================================================================
// RECENT ACTIVITY COMPONENT
// =============================================================================

export const RecentActivity: React.FC<RecentActivityProps> = ({
  activities,
  maxItems = 5,
}) => {
  const displayedActivities = activities.slice(0, maxItems);

  return (
    <div className="bg-white rounded-xl border border-ink-200/50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-ink-100">
        <h3 className="font-semibold text-heading-sm text-ink-900 tracking-tight">
          Recent Activity
        </h3>
        <Link
          href="/activity"
          className="text-body-sm text-ink-500 hover:text-ink-700 transition-colors"
        >
          View all
        </Link>
      </div>

      {/* Activity List */}
      {displayedActivities.length > 0 ? (
        <div className="divide-y divide-ink-100">
          {displayedActivities.map((activity) => (
            <Link
              key={activity.id}
              href={activity.link || '#'}
              className="flex items-start gap-3 px-5 py-3.5 hover:bg-ink-50 transition-colors"
            >
              <div className="w-8 h-8 rounded-lg bg-ink-100 flex items-center justify-center text-ink-500 flex-shrink-0">
                {getActivityIcon(activity.type)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-body-sm font-medium text-ink-900 truncate">
                  {activity.title}
                </p>
                {activity.description && (
                  <p className="text-body-xs text-ink-500 truncate mt-0.5">
                    {activity.description}
                  </p>
                )}
              </div>
              <span className="text-body-xs text-ink-400 flex-shrink-0">
                {formatTime(activity.timestamp)}
              </span>
            </Link>
          ))}
        </div>
      ) : (
        <div className="px-5 py-12 text-center">
          <div className="w-12 h-12 rounded-xl bg-ink-100 flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6 text-ink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-body-sm text-ink-500">No recent activity</p>
          <p className="text-body-xs text-ink-400 mt-1">
            Your searches and analyses will appear here
          </p>
        </div>
      )}
    </div>
  );
};

export default RecentActivity;
