'use client';

// =============================================================================
// FORGOT PASSWORD PAGE
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';
import { forgotPasswordSchema } from '@/lib/validations/auth';
import { ZodError } from 'zod';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      // Validate email
      const validatedData = forgotPasswordSchema.parse({ email });

      // Call forgot password API
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validatedData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send reset email');
      }

      setIsSuccess(true);
    } catch (error) {
      if (error instanceof ZodError) {
        setError(error.errors[0]?.message || 'Invalid email');
      } else if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // Success state
  if (isSuccess) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
        {/* Success icon */}
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-success-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>

        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Check your email
        </h1>
        <p className="text-body-md text-neutral-600 mb-6">
          If an account with <strong className="text-navy-900">{email}</strong> exists, 
          we&apos;ve sent you a link to reset your password.
        </p>
        <p className="text-body-sm text-neutral-500 mb-8">
          Didn&apos;t receive the email? Check your spam folder or{' '}
          <button
            onClick={() => {
              setIsSuccess(false);
              setEmail('');
            }}
            className="text-navy-500 hover:text-navy-700 font-medium"
          >
            try again
          </button>
        </p>

        <Link
          href="/signin"
          className="inline-flex items-center justify-center h-12 px-6 bg-navy-500 text-white font-medium rounded-lg hover:bg-navy-600 transition-colors"
        >
          Back to sign in
        </Link>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-cream-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-navy-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
          </svg>
        </div>
        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Forgot your password?
        </h1>
        <p className="text-body-md text-neutral-600">
          No worries! Enter your email and we&apos;ll send you reset instructions.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 bg-error-50 border border-error-200 rounded-lg">
          <p className="text-body-sm text-error-700 flex items-center gap-2">
            <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {error}
          </p>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label htmlFor="email" className="block text-body-sm font-medium text-navy-700 mb-2">
            Email address
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              setError('');
            }}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${error 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="you@example.com"
          />
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="
            w-full h-12
            bg-terra-500 text-white
            font-heading font-medium
            rounded-lg
            transition-all duration-fast
            hover:bg-terra-600 hover:-translate-y-0.5 hover:shadow-terra
            focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2
            disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none
          "
        >
          {isSubmitting ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Sending...
            </span>
          ) : (
            'Send reset link'
          )}
        </button>
      </form>

      {/* Back to sign in */}
      <p className="mt-8 text-center">
        <Link
          href="/signin"
          className="inline-flex items-center gap-2 text-body-sm font-medium text-navy-500 hover:text-navy-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to sign in
        </Link>
      </p>
    </div>
  );
}
