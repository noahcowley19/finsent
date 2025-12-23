'use client';

// =============================================================================
// NOTIFICATIONS SETTINGS PAGE
// =============================================================================
// Email and notification preferences
//
// Location: frontend/app/settings/notifications/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { useAuth } from '@/lib/auth-context';

interface NotificationSetting {
  id: string;
  title: string;
  description: string;
  enabled: boolean;
  proOnly?: boolean;
}

const defaultSettings: NotificationSetting[] = [
  {
    id: 'price_alerts',
    title: 'Price Alerts',
    description: 'Get notified when stocks in your watchlist hit your target prices',
    enabled: true,
    proOnly: true,
  },
  {
    id: 'sentiment_shifts',
    title: 'Sentiment Shifts',
    description: 'Get notified when sentiment changes significantly for watched stocks',
    enabled: true,
    proOnly: true,
  },
  {
    id: 'insider_activity',
    title: 'Insider Activity',
    description: 'Get notified when insiders buy or sell stocks in your watchlist',
    enabled: false,
    proOnly: true,
  },
  {
    id: 'weekly_digest',
    title: 'Weekly Digest',
    description: 'Receive a weekly summary of your portfolio and watchlist performance',
    enabled: true,
  },
  {
    id: 'product_updates',
    title: 'Product Updates',
    description: 'Stay informed about new features and improvements',
    enabled: true,
  },
  {
    id: 'tips_tutorials',
    title: 'Tips & Tutorials',
    description: 'Receive helpful tips on how to get the most out of Caveray',
    enabled: false,
  },
];

export default function NotificationsSettingsPage() {
  const { user } = useAuth();
  const isPro = user?.tier === 'pro';
  
  const [settings, setSettings] = useState<NotificationSetting[]>(defaultSettings);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleToggle = (id: string) => {
    setSettings((prev) =>
      prev.map((setting) =>
        setting.id === id ? { ...setting, enabled: !setting.enabled } : setting
      )
    );
  };

  const handleSave = async () => {
    setIsSaving(true);
    setMessage(null);

    try {
      // TODO: Implement API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setMessage({ type: 'success', text: 'Notification preferences saved' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save preferences' });
    } finally {
      setIsSaving(false);
    }
  };

  const alertSettings = settings.filter((s) => s.proOnly);
  const emailSettings = settings.filter((s) => !s.proOnly);

  return (
    <div className="space-y-8">
      {/* Alert notifications (Pro only) */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="font-heading font-semibold text-heading-md text-navy-900">
              Alert Notifications
            </h2>
            <p className="text-body-sm text-neutral-500 mt-1">
              Real-time alerts for your watchlist
            </p>
          </div>
          {!isPro && (
            <span className="px-2 py-1 bg-terra-100 text-terra-700 text-caption font-medium rounded-full">
              Pro Feature
            </span>
          )}
        </div>

        <div className="space-y-4">
          {alertSettings.map((setting) => (
            <div
              key={setting.id}
              className={`flex items-center justify-between p-4 rounded-lg ${
                !isPro ? 'bg-cream-50 opacity-60' : 'bg-cream-50'
              }`}
            >
              <div className="flex-1 pr-4">
                <p className="text-body-sm font-medium text-navy-900">{setting.title}</p>
                <p className="text-caption text-neutral-500">{setting.description}</p>
              </div>
              <button
                onClick={() => isPro && handleToggle(setting.id)}
                disabled={!isPro}
                className={`
                  relative w-12 h-7 rounded-full transition-colors duration-fast
                  ${!isPro ? 'cursor-not-allowed' : 'cursor-pointer'}
                  ${setting.enabled && isPro ? 'bg-terra-500' : 'bg-neutral-300'}
                `}
              >
                <span
                  className={`
                    absolute top-0.5 w-6 h-6 rounded-full bg-white shadow-sm
                    transition-transform duration-fast
                    ${setting.enabled && isPro ? 'left-5' : 'left-0.5'}
                  `}
                />
              </button>
            </div>
          ))}
        </div>

        {!isPro && (
          <p className="mt-4 text-body-sm text-neutral-500">
            <a href="/pricing" className="text-terra-500 hover:text-terra-600 font-medium">
              Upgrade to Pro
            </a>{' '}
            to enable real-time alert notifications.
          </p>
        )}
      </div>

      {/* Email notifications */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="mb-6">
          <h2 className="font-heading font-semibold text-heading-md text-navy-900">
            Email Notifications
          </h2>
          <p className="text-body-sm text-neutral-500 mt-1">
            Manage what emails you receive from Caveray
          </p>
        </div>

        <div className="space-y-4">
          {emailSettings.map((setting) => (
            <div
              key={setting.id}
              className="flex items-center justify-between p-4 rounded-lg bg-cream-50"
            >
              <div className="flex-1 pr-4">
                <p className="text-body-sm font-medium text-navy-900">{setting.title}</p>
                <p className="text-caption text-neutral-500">{setting.description}</p>
              </div>
              <button
                onClick={() => handleToggle(setting.id)}
                className={`
                  relative w-12 h-7 rounded-full transition-colors duration-fast cursor-pointer
                  ${setting.enabled ? 'bg-terra-500' : 'bg-neutral-300'}
                `}
              >
                <span
                  className={`
                    absolute top-0.5 w-6 h-6 rounded-full bg-white shadow-sm
                    transition-transform duration-fast
                    ${setting.enabled ? 'left-5' : 'left-0.5'}
                  `}
                />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Message */}
      {message && (
        <div
          className={`p-4 rounded-lg ${
            message.type === 'success'
              ? 'bg-success-50 text-success-700'
              : 'bg-error-50 text-error-700'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Save button */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="px-6 py-2.5 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors disabled:opacity-50"
        >
          {isSaving ? 'Saving...' : 'Save Preferences'}
        </button>
      </div>
    </div>
  );
}
