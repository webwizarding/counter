import sys
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

sys.path.append('OpenCV-Playing-Card-Detector-master')

import Cards
import VideoStream

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    return thresh

def detect_cards(frame):
    preprocessed_frame = preprocess_frame(frame)
    card_contours, cards = Cards.find_cards(preprocessed_frame)
    recognized_cards = []

    if len(cards) != 0:
        for card in cards:
            best_rank_match, best_suit_match, rank_diff, suit_diff = Cards.match_card(card)
            card.rank = best_rank_match
            card.suit = best_suit_match
            recognized_cards.append(card)

    return recognized_cards

def calculate_hand_total(cards):
    total = 0
    aces = 0
    for card in cards:
        if card.rank in ['J', 'Q', 'K', '10']:
            total += 10
        elif card.rank == 'A':
            aces += 1
        else:
            total += int(card.rank)

    for _ in range(aces):
        if total + 11 <= 21:
            total += 11
        else:
            total += 1
    return total

def count_cards(cards, running_count):
    card_values = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                   '7': 0, '8': 0, '9': 0,
                   '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}
    for card in cards:
        if card.rank in card_values:
            running_count += card_values[card.rank]
    return running_count

def make_decision(hand_total, true_count):
    if hand_total < 17 or (hand_total < 21 and true_count > 1):
        return 'Hit'
    else:
        return 'Stay'

def adjust_betting(true_count, base_bet):
    if true_count > 1:
        return base_bet * 2
    elif true_count < -1:
        return base_bet // 2
    else:
        return base_bet

def cluster_hands(cards):
    if not cards:
        return []

    card_centers = np.array([card.center for card in cards])
    clustering = DBSCAN(eps=100, min_samples=1).fit(card_centers)
    labels = clustering.labels_

    clustered_hands = []
    for label in set(labels):
        hand = [cards[i] for i in range(len(cards)) if labels[i] == label]
        clustered_hands.append(hand)

    return clustered_hands

# Main Function
def main():
    cap = initialize_camera()
    running_count = 0
    num_decks = 6
    base_bet = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cards = detect_cards(frame)
        hand_clusters = cluster_hands(cards)  

        for hand in hand_clusters:
            hand_total = calculate_hand_total(hand)
            running_count = count_cards(hand, running_count)
            true_count = running_count / num_decks

            decision = make_decision(hand_total, true_count)
            bet = adjust_betting(true_count, base_bet)

            x, y = hand[0].center  # first card's center for text
            cv2.putText(frame, f'Decision: {decision}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Bet: ${bet}', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Blackjack Helper', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
