def get_avatar_response(style_name, score):
    """
    Tailors the Avatar's dialogue based on how much the patron liked the style.
    """
    if score > 0.8:
        return f"Wow! I saw that smile. The {style_name} looks absolutely stunning on you. I've already sent the specs to the stylist!"
    elif score > 0.4:
        return f"I think we found a winner! The {style_name} really complements your features. What do you think?"
    elif score > 0.1:
        return f"The {style_name} is a solid choice, though I noticed you were still considering your options. Should I show the stylist this one?"
    else:
        return f"I've saved the {style_name} for you, but let's keep looking if you're not feeling it. Your perfect look is still in here!"

# Example usage for the iPhone frontend:
# avatar_text = get_avatar_response("Platinum Waves", 0.92)
