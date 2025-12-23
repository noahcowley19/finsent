'use client';

// =============================================================================
// BILLING SETTINGS PAGE
// =============================================================================
// Subscription and payment management
//
// Location: frontend/app/settings/billing/page.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';

export default function BillingSettingsPage() {
  const { user } = useAuth();
  const isPro = user?.tier === 'pro';

  // Mock data - replace with real data
  const subscription = isPro
    ? {
        plan: 'Pro',
        price: '$9.99/month',
        renewalDate: 'January 15, 2025',
        paymentMethod: '**** **** **** 4242',
      }
    : null;

  const invoices = [
    { id: 'INV-001', date: 'Dec 15, 2024', amount: '$9.99', status: 'Paid' },
    { id: 'INV-002', date: 'Nov 15, 2024', amount: '$9.99', status: 'Paid' },
    { id: 'INV-003', date: 'Oct 15, 2024', amount: '$9.99', status: 'Paid' },
  ];

  return (
    <div className="space-y-8">
      {/* Current plan */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
          Current Plan
        </h2>

        {isPro ? (
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="font-heading font-semibold text-xl text-navy-900">
                  Pro Plan
                </span>
                <span className="px-2 py-0.5 bg-terra-100 text-terra-700 text-caption font-medium rounded-full">
                  Active
                </span>
              </div>
              <p className="text-body-md text-neutral-600 mb-4">
                {subscription?.price} • Renews on {subscription?.renewalDate}
              </p>
              <ul className="space-y-2 text-body-sm text-neutral-600">
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Unlimited analyses
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Quant Lab access
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Priority support
                </li>
              </ul>
            </div>
            <div className="space-y-2">
              <button className="w-full px-4 py-2 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors">
                Change Plan
              </button>
              <button className="w-full px-4 py-2 text-body-sm font-medium text-error-600 hover:text-error-700 transition-colors">
                Cancel Subscription
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
              <svg className="w-8 h-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="font-heading font-semibold text-navy-900 mb-2">
              You&apos;re on the Free Plan
            </h3>
            <p className="text-body-md text-neutral-600 mb-6 max-w-md mx-auto">
              Upgrade to Pro for unlimited analyses, Quant Lab access, and more.
            </p>
            <Link
              href="/pricing"
              className="inline-flex items-center gap-2 px-6 py-3 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Upgrade to Pro
            </Link>
          </div>
        )}
      </div>

      {/* Payment method */}
      {isPro && (
        <div className="bg-white rounded-xl border border-border-light p-6">
          <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
            Payment Method
          </h2>

          <div className="flex items-center justify-between p-4 bg-cream-50 rounded-lg">
            <div className="flex items-center gap-4">
              <div className="w-12 h-8 bg-navy-900 rounded flex items-center justify-center">
                <span className="text-white text-caption font-bold">VISA</span>
              </div>
              <div>
                <p className="text-body-sm font-medium text-navy-900">
                  {subscription?.paymentMethod}
                </p>
                <p className="text-caption text-neutral-500">Expires 12/2026</p>
              </div>
            </div>
            <button className="px-4 py-2 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors">
              Update
            </button>
          </div>
        </div>
      )}

      {/* Billing history */}
      {isPro && invoices.length > 0 && (
        <div className="bg-white rounded-xl border border-border-light p-6">
          <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
            Billing History
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border-light">
                  <th className="text-left py-3 text-body-sm font-medium text-neutral-500">Invoice</th>
                  <th className="text-left py-3 text-body-sm font-medium text-neutral-500">Date</th>
                  <th className="text-left py-3 text-body-sm font-medium text-neutral-500">Amount</th>
                  <th className="text-left py-3 text-body-sm font-medium text-neutral-500">Status</th>
                  <th className="text-right py-3 text-body-sm font-medium text-neutral-500">Action</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((invoice) => (
                  <tr key={invoice.id} className="border-b border-border-light last:border-0">
                    <td className="py-4 text-body-sm font-medium text-navy-900">{invoice.id}</td>
                    <td className="py-4 text-body-sm text-neutral-600">{invoice.date}</td>
                    <td className="py-4 text-body-sm text-neutral-600">{invoice.amount}</td>
                    <td className="py-4">
                      <span className="px-2 py-1 bg-success-100 text-success-700 text-caption font-medium rounded-full">
                        {invoice.status}
                      </span>
                    </td>
                    <td className="py-4 text-right">
                      <button className="text-body-sm text-navy-500 hover:text-navy-700 transition-colors">
                        Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
