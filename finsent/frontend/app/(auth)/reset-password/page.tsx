'use client';

// =============================================================================
// RESET PASSWORD PAGE
// =============================================================================

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { resetPasswordSchema, calculatePasswordStrength } from '@/lib/validations/auth';
import { ZodError } from 'zod';

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get('token');

  const [formData, setFormData] = useState({
    password: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isValidToken, setIsValidToken] = useState<boolean | null>(null);
  const [tokenError, setTokenError] = useState('');
  const [passwordStrength, setPasswordStrength] = useState({ score: 0, label: 'Very Weak', suggestions: [] as string[] });

  // Verify token on mount
  useEffect(() => {
    if (!token) {
      setIsValidToken(false);
      setTokenError('No reset token provided');
      return;
    }

    const verifyToken = async () => {
      try {
        const response = await fetch(`/api/auth/reset-password?token=${token}`);
        const data = await response.json();
        
        if (data.valid) {
          setIsValidToken(true);
        } else {
          setIsValidToken(false);
          setTokenError(data.error || 'Invalid or expired reset link');
        }
      } catch {
        setIsValidToken(false);
        setTokenError('Failed to verify reset link');
      }
    };

    verifyToken();
  }, [token]);

  // Update password strength
  useEffect(() => {
    if (formData.password) {
      setPasswordStrength(calculatePasswordStrength(formData.password));
    } else {
      setPasswordStrength({ score: 0, label: 'Very Weak', suggestions: [] });
    }
  }, [formData.password]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});
    setIsSubmitting(true);

    try {
      // Validate form data
      const validatedData = resetPasswordSchema.parse({
        token,
        ...formData,
      });

      // Call reset password API
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validatedData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to reset password');
      }

      setIsSuccess(true);
    } catch (error) {
      if (error instanceof ZodError) {
        const fieldErrors: Record<string, string> = {};
        error.errors.forEach((err) => {
          if (err.path[0]) {
            fieldErrors[err.path[0] as string] = err.message;
          }
        });
        setErrors(fieldErrors);
      } else if (error instanceof Error) {
        setErrors({ general: error.message });
      } else {
        setErrors({ general: 'An unexpected error occurred' });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const strengthColors = ['bg-neutral-200', 'bg-error-500', 'bg-warning-500', 'bg-success-400', 'bg-success-500'];

  // Loading state
  if (isValidToken === null) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
        <div className="w-8 h-8 mx-auto border-2 border-cream-200 border-t-navy-500 rounded-full animate-spin" />
        <p className="mt-4 text-body-md text-neutral-600">Verifying reset link...</p>
      </div>
    );
  }

  // Invalid token state
  if (!isValidToken) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-error-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Invalid Reset Link
        </h1>
        <p className="text-body-md text-neutral-600 mb-8">
          {tokenError}
        </p>
        <Link
          href="/forgot-password"
          className="inline-flex items-center justify-center h-12 px-6 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
        >
          Request new link
        </Link>
      </div>
    );
  }

  // Success state
  if (isSuccess) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-success-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Password Reset Successfully
        </h1>
        <p className="text-body-md text-neutral-600 mb-8">
          Your password has been changed. You can now sign in with your new password.
        </p>
        <Link
          href="/signin"
          className="inline-flex items-center justify-center h-12 px-6 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
        >
          Sign in
        </Link>
      </div>
    );
  }

  // Reset form
  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-cream-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-navy-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Set new password
        </h1>
        <p className="text-body-md text-neutral-600">
          Please enter your new password below.
        </p>
      </div>

      {/* General error */}
      {errors.general && (
        <div className="mb-6 p-4 bg-error-50 border border-error-200 rounded-lg">
          <p className="text-body-sm text-error-700 flex items-center gap-2">
            <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {errors.general}
          </p>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Password field */}
        <div>
          <label htmlFor="password" className="block text-body-sm font-medium text-navy-700 mb-2">
            New password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="new-password"
            value={formData.password}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.password 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="Create a strong password"
          />
          {errors.password && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.password}</p>
          )}
          
          {/* Password strength indicator */}
          {formData.password && (
            <div className="mt-2">
              <div className="flex gap-1 mb-1">
                {[1, 2, 3, 4].map((level) => (
                  <div
                    key={level}
                    className={`h-1 flex-1 rounded-full transition-colors ${
                      level <= passwordStrength.score ? strengthColors[passwordStrength.score] : 'bg-neutral-200'
                    }`}
                  />
                ))}
              </div>
              <p className="text-caption text-neutral-500">
                Password strength: {passwordStrength.label}
              </p>
            </div>
          )}
        </div>

        {/* Confirm password field */}
        <div>
          <label htmlFor="confirmPassword" className="block text-body-sm font-medium text-navy-700 mb-2">
            Confirm new password
          </label>
          <input
            id="confirmPassword"
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={formData.confirmPassword}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.confirmPassword 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="Confirm your new password"
          />
          {errors.confirmPassword && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.confirmPassword}</p>
          )}
        </div>

        {/* Submit button */}
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
              Resetting password...
            </span>
          ) : (
            'Reset password'
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
