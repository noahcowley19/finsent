'use client';

// =============================================================================
// RECENT ACTIVITY COMPONENT
// =============================================================================
// Displays a feed of recent user activities
//
// Location: frontend/components/dashboard/RecentActivity.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export interface Activity {
  id: string;
  type: 'search' | 'analysis' | 'portfolio' | 'watchlist' | 'alert';
  title: string;
  description: string;
  timestamp: Date;
  link?: string;
}

export interface RecentActivityProps {
  /** List of activities */
  activities: Activity[];
  /** Loading state */
  isLoading?: boolean;
  /** Max items to show */
  limit?: number;
}

const activityIcons: Record<Activity['type'], React.ReactNode> = {
  search: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  ),
  analysis: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  portfolio: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
  ),
  watchlist: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  alert: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
    </svg>
  ),
};

const activityColors: Record<Activity['type'], string> = {
  search: 'bg-navy-100 text-navy-600',
  analysis: 'bg-terra-100 text-terra-600',
  portfolio: 'bg-success-100 text-success-600',
  watchlist: 'bg-warning-100 text-warning-600',
  alert: 'bg-error-100 text-error-600',
};

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) return 'Just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`;
  return date.toLocaleDateString();
}

export const RecentActivity: React.FC<RecentActivityProps> = ({
  activities,
  isLoading = false,
  limit = 5,
}) => {
  const displayedActivities = activities.slice(0, limit);

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="w-32 h-6 rounded bg-cream-100 animate-pulse" />
          <div className="w-16 h-4 rounded bg-cream-100 animate-pulse" />
        </div>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="flex items-start gap-4 animate-pulse">
              <div className="w-8 h-8 rounded-lg bg-cream-100" />
              <div className="flex-1">
                <div className="w-48 h-4 rounded bg-cream-100 mb-2" />
                <div className="w-32 h-3 rounded bg-cream-100" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Recent Activity
        </h3>
        <Link
          href="/activity"
          className="text-body-sm text-navy-500 hover:text-navy-700 transition-colors"
        >
          View all
        </Link>
      </div>

      {/* Activity list */}
      {displayedActivities.length === 0 ? (
        <div className="text-center py-8">
          <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
            <svg className="w-6 h-6 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-body-sm text-neutral-500">No recent activity</p>
          <p className="text-caption text-neutral-400 mt-1">
            Start by searching for a stock
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {displayedActivities.map((activity) => (
            <div
              key={activity.id}
              className="flex items-start gap-4 group"
            >
              {/* Icon */}
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${activityColors[activity.type]}`}>
                {activityIcons[activity.type]}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                {activity.link ? (
                  <Link
                    href={activity.link}
                    className="text-body-sm font-medium text-navy-900 hover:text-navy-700 transition-colors truncate block"
                  >
                    {activity.title}
                  </Link>
                ) : (
                  <p className="text-body-sm font-medium text-navy-900 truncate">
                    {activity.title}
                  </p>
                )}
                <p className="text-caption text-neutral-500 truncate">
                  {activity.description}
                </p>
              </div>

              {/* Timestamp */}
              <span className="text-caption text-neutral-400 flex-shrink-0">
                {formatTimeAgo(activity.timestamp)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RecentActivity;
